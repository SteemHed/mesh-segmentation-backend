from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import trimesh
import io

app = FastAPI(title="Free Cloud 3D Mesh API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        loaded = trimesh.load(io.BytesIO(contents), file_type=file_ext)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        sub_meshes = mesh.split(only_watertight=False)

        parts_metadata = []
        global_centroid = mesh.centroid
        global_height = mesh.extents[2]

        for idx, sub in enumerate(sub_meshes):
            if len(sub.faces) < 10:
                continue

            bounds = sub.bounding_box.extents
            centroid = sub.centroid
            rel_z = (centroid[2] - global_centroid[2]) / (global_height / 2.0 if global_height > 0 else 1.0)
            aspect_ratio = max(bounds) / (min(bounds) + 1e-6)

            if aspect_ratio > 3.0:
                category = "Elongated_Part"
            elif rel_z > 0.4:
                category = "Top_Section"
            elif rel_z < -0.4:
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
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")
