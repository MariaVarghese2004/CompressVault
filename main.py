from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import io
import os
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compress_jpeg_image(input_image: Image.Image, output_io: io.BytesIO, quality=70):
    """Compress JPEG image by removing metadata and adjusting quality."""
    input_image = input_image.convert('RGB')  # Ensure no metadata
    output_io.seek(0)  # Reset stream position before saving
    output_io.truncate()  # Clear previous data
    input_image.save(output_io, 'JPEG', quality=quality, optimize=True)

def get_file_size(file_io: io.BytesIO):
    """Return the size of the file in bytes from BytesIO."""
    file_io.seek(0, io.SEEK_END)
    size = file_io.tell()
    file_io.seek(0)  # Reset stream position
    return size

def compress_and_convert(input_image: Image.Image, jpeg_quality=75, original_size_kb=None):
    """Convert PNG to JPEG, compress, and then convert back to PNG. Ensure final size is not larger than the original."""
    temp_jpeg_io = io.BytesIO()
    final_png_io = io.BytesIO()
    
    # Convert PNG to JPEG with compression
    input_image = input_image.convert('RGB')  # Ensure image is in RGB mode for JPEG
    compress_jpeg_image(input_image, temp_jpeg_io, quality=jpeg_quality)
    
    # Convert JPEG back to PNG
    temp_jpeg_io.seek(0)  # Reset stream position before opening
    with Image.open(temp_jpeg_io) as jpeg_image:
        jpeg_image.save(final_png_io, format='PNG', optimize=True)
    
    final_png_io.seek(0)  # Reset stream position before checking size
    
    # Check file size and adjust if necessary
    if original_size_kb is not None:
        final_size_kb = get_file_size(final_png_io) / 1024
        if final_size_kb > original_size_kb:
            # Reduce PNG quality if the size increased
            num_colors = 256
            compress_level = 9
            final_png_io = io.BytesIO()
            compress_png_image(input_image, final_png_io, max_size_kb=original_size_kb, compress_level=compress_level, num_colors=num_colors)
    
    final_png_io.seek(0)  # Reset stream position before returning
    return final_png_io



def compress_image(input_image: Image.Image, output_io: io.BytesIO, max_size_kb, initial_quality=85, min_quality=40):
    """Compress image and adjust quality to ensure size is under max_size_kb without resizing."""
    quality = initial_quality
    while quality > min_quality:
        compress_jpeg_image(input_image, output_io, quality)
        output_size = get_file_size(output_io)
        if output_size <= max_size_kb * 1024:
            break
        quality -= 5

def compress_png_image(input_image: Image.Image, output_io: io.BytesIO, max_size_kb, compress_level=9, num_colors=256):
    """Compress PNG image with optimization while maintaining quality."""
    while compress_level >= 0:
        # Convert to palette mode with a valid number of colors (1-256)
        if input_image.mode in ('RGBA', 'RGB'):
            input_image = input_image.convert('P', palette=Image.ADAPTIVE, colors=min(num_colors, 256))
        
        # Save the image with lossless compression and optimization
        output_io.seek(0)  # Reset stream position before saving
        output_io.truncate()  # Clear previous data
        input_image.save(output_io, 'PNG', optimize=True, compress_level=compress_level)
        
        output_size = get_file_size(output_io)
        if output_size <= max_size_kb * 1024:
            break
        compress_level -= 1

def calculate_image_similarity(image1: Image.Image, image2: Image.Image):
    image1 = image1.convert('L')
    image2 = image2.convert('L')
    img1 = np.array(image1)
    img2 = np.array(image2)
    similarity_index, _ = ssim(img1, img2, full=True)
    return similarity_index * 100                                                                                                 


@app.post("/get-image-stats/")
async def get_image_stats(file: UploadFile = File(...)):
    try:
        # Read file content once
        file_content = await file.read()
        original_io = io.BytesIO(file_content)
        image = Image.open(original_io)
        original_size_kb = len(file_content) / 1024  # Calculate original size from file content

        # Reinitialize the BytesIO for further processing
        original_io.seek(0)
        output_io = io.BytesIO()
        file_extension = os.path.splitext(file.filename)[-1].lower()

        if file_extension in ['.jpeg', '.jpg']:
            # Compress JPEG image
            compress_image(image, output_io, max_size_kb=70)
            media_type = 'image/jpeg'
        
        elif file_extension == '.png':
            final_png_io = compress_and_convert(image, jpeg_quality=95, original_size_kb=original_size_kb)
            final_png_io.seek(0)
            output_io = final_png_io
            media_type = 'image/png'
        
        elif file_extension == '.tiff':
            image = image.convert('P', palette=Image.ADAPTIVE, colors=256)
            image.save(output_io, format='TIFF', compression='tiff_lzw')
            media_type = 'image/tiff'
        
        elif file_extension == '.bmp':
            image = image.convert('P', palette=Image.ADAPTIVE, colors=256)
            image.save(output_io, format='BMP')
            media_type = 'image/bmp'
        
        else:
            return JSONResponse(content={"message": "Only JPEG, PNG, TIFF, and BMP formats are supported"})

        # Calculate compressed file size
        output_io.seek(0)
        compressed_size_kb = get_file_size(output_io) / 1024

        # Calculate similarity percentage
        compressed_image = Image.open(output_io)
        similarity_percentage = calculate_image_similarity(image, compressed_image)

        # Prepare the response
        return JSONResponse(content={
            "original_file_size_kb": original_size_kb,
            "compressed_file_size_kb": compressed_size_kb,
            "similarity_percentage": similarity_percentage
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)})


ALLOWED_FORMATS = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp']

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Load the image from the uploaded file
        image = Image.open(io.BytesIO(await file.read()))
        original_size_kb = get_file_size(io.BytesIO(await file.read())) / 1024  # Get original file size

        output_io = io.BytesIO()
        file_extension = os.path.splitext(file.filename)[-1].lower()

        if file_extension in ['.jpeg', '.jpg']:
            # Compress JPEG image
            compress_image(image, output_io, max_size_kb=70)
            media_type = 'image/jpeg'
        
        elif file_extension == '.png':
            final_png_io = compress_and_convert(image, jpeg_quality=95, original_size_kb=original_size_kb)
            final_png_io.seek(0)
            response = StreamingResponse(final_png_io, media_type='image/png')
            response.headers["Content-Disposition"] = f"attachment; filename={file.filename}"
            return response
        
        elif file_extension == '.tiff':
            image = image.convert('P', palette=Image.ADAPTIVE, colors=256)
            image.save(output_io, format='TIFF', compression='tiff_lzw')
        
        elif file_extension == '.bmp':
            image = image.convert('P', palette=Image.ADAPTIVE, colors=256)
            image.save(output_io, format='BMP')
        
        else:
            return JSONResponse(content={"message": "Only JPEG, PNG, TIFF, and BMP formats are supported"})
        
        output_io.seek(0)
        compressed_image = Image.open(output_io)
        similarity_percentage = calculate_image_similarity(image, compressed_image)

        if similarity_percentage >= 80:
         output_io.seek(0)
         response = StreamingResponse(output_io, media_type=media_type)
         response.headers["Content-Disposition"] = f"attachment; filename={file.filename}"
         return response
        else:
            return JSONResponse(content={"message": "Image similarity is less than 80%."})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
