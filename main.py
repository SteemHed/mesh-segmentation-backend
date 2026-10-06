from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import trimesh
import io
import json
import gc  # Garbage collector for memory management

app = FastAPI(title="Free Cloud 3D Mesh API")

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
    max_faces: int = Form(default=25000)
):
    # Free any lingering memory before processing new request
    gc.collect()

    file_ext = file.filename.split('.')[-1].lower()
    
    # 1. Read file bytes with safety check
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    print(f"Received file: {file.filename} ({file_size_mb:.2f} MB), max_faces target: {max_faces}")

    # Guardrail: Reject massive files over 45MB to prevent 512MB RAM crash
    if file_size_mb > 45:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large ({file_size_mb:.1f}MB). The free cloud tier supports files up to 45MB. Please compress or decimate the model locally first."
        )

    try:
        # 2. Load mesh safely from buffer
        loaded = trimesh.load(io.BytesIO(contents), file_type=file_ext)
        del contents  # Immediately free raw byte memory
        gc.collect()

        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

        print(f"Mesh parsed successfully: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # 3. Downsample dense geometry using memory-safe voxelization
        if max_faces > 0 and len(mesh.faces) > max_faces:
            print(f"Downsampling from {len(mesh.faces)} to target budget {max_faces}...")
            bounding_extent = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
            pitch_divisor = (max_faces / 1000.0) * 3.5
            pitch = bounding_extent / max(pitch_divisor, 25.0)
            
            mesh = mesh.voxelized(pitch=pitch).marching_cubes
            gc.collect()

        # 4. Geometric Component Splitting
        sub_meshes = mesh.split(only_watertight=False)
        print(f"Identified {len(sub_meshes)} raw sub-components")

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
            
            # Set metadata and node name for GLTF export
            sub.metadata["name"] = part_name
            scene.add_geometry(sub, node_name=part_name)

            parts_metadata.append({
                "id": idx + 1,
                "name": part_name,
                "vertices": len(sub.vertices),
                "faces": len(sub.faces)
            })

        # 5. Export binary GLB byte stream
        glb_bytes = scene.export(file_type="glb")
        gc.collect()

        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "X-Parts-Metadata": json.dumps(parts_metadata)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing mesh: {str(e)}")
        raise HTTPException(status_code=400, detail=f"3D Engine Error: {str(e)}")
