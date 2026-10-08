from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import trimesh
import io
import json
import gc

app = FastAPI(title="Free Cloud 3D Mesh API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Parts-Metadata"]
)

@app.get("/")
def health_check():
    return {"status": "online"}

@app.options("/segment")
async def options_segment():
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "X-Parts-Metadata"
        }
    )

@app.post("/segment")
async def segment_mesh(
    file: UploadFile = File(...),
    max_faces: int = Query(default=25000)  # Changed to Query parameter for cleaner stream uploads
):
    gc.collect()
    file_ext = file.filename.split('.')[-1].lower()
    contents = await file.read()
    
    try:
        # Load mesh from buffer
        loaded = trimesh.load(io.BytesIO(contents), file_type=file_ext)
        del contents
        gc.collect()

        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

        # Downsample if dense
        if max_faces > 0 and len(mesh.faces) > max_faces:
            bounding_extent = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
            pitch_divisor = (max_faces / 1000.0) * 3.5
            pitch = bounding_extent / max(pitch_divisor, 25.0)
            mesh = mesh.voxelized(pitch=pitch).marching_cubes
            gc.collect()

        # Component splitting
        sub_meshes = mesh.split(only_watertight=False)
        scene = trimesh.Scene()
        parts_metadata = []
        global_centroid = mesh.centroid
        global_height = mesh.extents[2] if mesh.extents[2] > 0 else 1.0

        for idx, sub in enumerate(sub_meshes):
            if len(sub.faces) < 12:
                continue

            bounds = sub.bounding_box.extents
            centroid = sub.centroid
            rel_z = (centroid[2] - global_centroid[2]) / (global_height / 2.0)
            aspect_ratio = max(bounds) / (min(bounds) + 1e-6)

            if aspect_ratio > 3.0:
                category = "Elongated_Part"
            elif rel_z > 0.35:
                category = "Top_Section"
            elif rel_z < -0.35:
                category = "Base_Section"
            else:
                category = "Mid_Section"

            part_name = f"{category}_{idx + 1}"
            sub.metadata["name"] = part_name
            scene.add_geometry(sub, node_name=part_name)

            parts_metadata.append({
                "id": idx + 1,
                "name": part_name,
                "vertices": len(sub.vertices),
                "faces": len(sub.faces)
            })

        glb_bytes = scene.export(file_type="glb")
        gc.collect()

        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "X-Parts-Metadata": json.dumps(parts_metadata),
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "X-Parts-Metadata"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
