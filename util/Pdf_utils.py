from os.path import exists
import fitz
import os
# to load pdf and image that it contains

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

pdf_dir = r"C:\Users\harsh\Desktop\Projects\langchain\Source"
output_image_dir = r"C:\Users\harsh\Desktop\Projects\langchain\output_image"
if not exists(output_image_dir) :
    os.mkdir(output_image_dir)

#  Load text using langchain PyMuPDFLoader
loader = PyMuPDFLoader(fr"{pdf_dir}\Class_10\Math\jemh1a1.pdf")
pdf_doc = loader.load()

# Extract images using PyMuPDFLoader and save them in output_image_dir
doc = fitz.open(fr"{pdf_dir}\Class_10\Math\jemh1a1.pdf")
image_metadata = []
image_mapping={}
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    image_list = doc.get_page_images(page_num)
    page_images = []
    for img_idx, img in enumerate(image_list):
        xref = img[0] # Image reference
        pix = fitz.Pixmap(doc,xref)
        img_filename = f"page_{page_num+1}_img_{img_idx + 1}.png"
        img_path = os.path.join(output_image_dir, img_filename)

        if pix.n<5:
            pix.save(img_path)
        else:
            # Convert CMYK to RGM to use it for screen
            pix = fitz.Pixmap(fitz.csRGB,pix)
            pix.save(img_path)
        image_metadata.append({
            "page":page_num+1,
            "image_path":img_path
        })
        page_images.append(img_path)
    if page_images:
        image_mapping[page_num+1]= page_images

doc.close()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_doc = text_splitter.split_documents(pdf_doc)

documentWithMetadata = []
for doc in chunked_doc:
    document = Document(page_content=doc.page_content,
                        metadata= {"page_num":doc.metadata.get("page", None)})
    documentWithMetadata.append(document)
vector_store = FAISS.from_documents(documentWithMetadata,embeddings)

def retriever_with_images(query):

    retriever = vector_store.as_retriever()
    result = retriever.invoke(query, k=3)
    final_result=[]
    for doc in result:

        text = doc.page_content
        page_num = doc.metadata.get("page", None)
        response= {"text":text}
        if any(kw in text.lower() for kw in ["see figure","as shown in image", "see image for refrence", "illustration", "diagram"]):
            if page_num in image_mapping.keys():
                response["images"] = image_mapping[page_num]
        final_result.append(response)
    return final_result

query = "explain proofs of mathematics"
results = retriever_with_images(query)

for res in results:
    print(f"Text: {res['text']}\n")
    if "images" in res:
        print(f"Relevant Images: {res['images']}\n")
