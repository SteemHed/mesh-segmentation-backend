from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import trimesh
import io
import json

app = FastAPI(title="Free Cloud 3D Mesh API")

# Enable CORS and expose custom metadata headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Parts-Metadata"]
)

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/segment")
async def segment_mesh(
    file: UploadFile = File(...),
    max_faces: int = Form(40000)
):
    contents = await file.read()
    file_ext = file.filename.split('.')[-1].lower()

    try:
        # 1. Load raw mesh buffer
        loaded = trimesh.load(io.BytesIO(contents), file_type=file_ext)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

        # 2. Downsample if mesh exceeds target face budget
        if max_faces > 0 and len(mesh.faces) > max_faces:
            print(f"Decimating mesh from {len(mesh.faces)} to {max_faces} faces...")
            bounding_extent = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
            pitch_divisor = (max_faces / 1000.0) * 3.0
            pitch = bounding_extent / max(pitch_divisor, 30.0)
            mesh = mesh.voxelized(pitch=pitch).marching_cubes

        # 3. Perform geometric component splitting
        sub_meshes = mesh.split(only_watertight=False)

        scene = trimesh.Scene()
        parts_metadata = []
        global_centroid = mesh.centroid
        global_height = mesh.extents[2] if mesh.extents[2] > 0 else 1.0

        for idx, sub in enumerate(sub_meshes):
            if len(sub.faces) < 10:
                continue

            bounds = sub.bounding_box.extents
            centroid = sub.centroid
            rel_z = (centroid[2] - global_centroid[2]) / (global_height / 2.0)
            aspect_ratio = max(bounds) / (min(bounds) + 1e-6)

            # Spatial Heuristic Classification
            if aspect_ratio > 3.0:
                category = "Elongated_Part"
            elif rel_z > 0.35:
                category = "Top_Section"
            elif rel_z < -0.35:
                category = "Base_Section"
            else:
                category = "Mid_Section"

            part_name = f"{category}_{idx + 1}"
            
            # Attach part name to Trimesh sub-geometry and add to Scene graph
            sub.metadata["name"] = part_name
            scene.add_geometry(sub, node_name=part_name)

            parts_metadata.append({
                "id": idx + 1,
                "name": part_name,
                "vertices": len(sub.vertices),
                "faces": len(sub.faces)
            })

        # 4. Export scene as binary GLTF (.glb) byte stream
        glb_bytes = scene.export(file_type="glb")

        # Return binary stream with metadata passed in custom header
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "X-Parts-Metadata": json.dumps(parts_metadata)
            }
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")
