from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import trimesh
import io

app = FastAPI(title="Free Cloud 3D Mesh API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/segment")
async def segment_mesh(file: UploadFile = File(...)):
    contents = await file.read()
    file_ext = file.filename.split('.')[-1].lower()

    try:
        # 1. Load mesh from buffer
        loaded = trimesh.load(io.BytesIO(contents), file_type=file_ext)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

        # 2. Optimization: Decimate ultra-dense meshes to stay within 512MB RAM
        MAX_FACES = 60000
        if len(mesh.faces) > MAX_FACES:
            print(f"Mesh too dense ({len(mesh.faces)} faces). Decimating to {MAX_FACES} faces for memory safety...")
            mesh = mesh.simplify_quadratic_decimation(MAX_FACES)

        # 3. Geometric Component Splitting
        sub_meshes = mesh.split(only_watertight=False)

        parts_metadata = []
        global_centroid = mesh.centroid
        global_height = mesh.extents[2] if mesh.extents[2] > 0 else 1.0

        for idx, sub in enumerate(sub_meshes):
            # Filter out tiny noise fragments
            if len(sub.faces) < 15:
                continue

            bounds = sub.bounding_box.extents
            centroid = sub.centroid
            rel_z = (centroid[2] - global_centroid[2]) / (global_height / 2.0)
            aspect_ratio = max(bounds) / (min(bounds) + 1e-6)

            # Spatial Heuristic Tags
            if aspect_ratio > 3.0:
                category = "Elongated_Part"
            elif rel_z > 0.35:
                category = "Top_Section"
            elif rel_z < -0.35:
                category = "Base_Section"
            else:
                category = "Mid_Section"

            part_name = f"{category}_{idx + 1}"
            parts_metadata.append({
                "id": idx + 1,
                "name": part_name,
                "vertices": len(sub.vertices),
                "faces": len(sub.faces)
            })

        return {
            "status": "success",
            "total_parts": len(parts_metadata),
            "parts": parts_metadata
        }

    except Exception as e:
        print(f"Error during mesh processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")
