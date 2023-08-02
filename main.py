from typing import Union

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from predict import predict_image

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
posts = [{
    'title': 'This is my first blog Post',
    'content': ["This server is provided using a full CI/CD pipeline integrating github actions and AWS Tools.",
                "Here you'll find a classification model published in huggingface.co which you can try it with your own."],
    'author': 'Bruno Martinez',
    'publication_date': '2023-08-03',
    'comments': [
        {}
    ],
    'status': 'published',
    'link_model': 'https://huggingface.co/google/vit-base-patch16-224'
}]

@app.get("/", response_class=HTMLResponse)
async def read_posts(request: Request):
    return templates.TemplateResponse("blog.html", {"request": request, 
                                                    "posts": posts})

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    prediction = predict_image(file)
    # do something with the file
    return {"message": prediction}