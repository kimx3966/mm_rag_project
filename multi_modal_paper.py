# -*- coding: utf-8 -*-
from pathlib import Path
from pprint import pprint

import os


from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode

from llama_index.multi_modal_llms.openai import OpenAIMultiModal

import fitz  # PyMuPDF

def extract_images(input_pdf_path, output_dir):
    # PDF 파일 열기
    pdf_document = fitz.open(pdf_path)

    os.makedirs(output_dir, exist_ok=True)

    # PDF에서 이미지를 추출
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)  # 각 페이지 로드
        image_list = page.get_images(full=True)     # 페이지에서 이미지 가져오기

        # 각 이미지 저장
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]  # 이미지 객체의 참조 번호
            base_image = pdf_document.extract_image(xref)  # 이미지 정보 추출
            image_bytes = base_image["image"]  # 이미지 바이트 데이터
            image_ext = base_image["ext"]      # 이미지 확장자 (예: png, jpeg 등)

            # 파일 저장 경로 설정
            image_filename = f"{output_dir}/page{page_number+1}_image{image_index}.{image_ext}"

            # 이미지 파일로 저장
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)

            print(f"이미지를 저장했습니다: {image_filename}")

    pdf_document.close()

pdf_dir = "./pdf_data/paper"
pdf_path = os.path.join(pdf_dir,"Attention is All you Need.pdf")

img_dir = "multimodal_data"
extract_images(input_pdf_path=pdf_path, output_dir=img_dir)


storage_context = StorageContext.from_defaults(
    persist_dir=pdf_dir
)
pdf_index = load_index_from_storage(storage_context)


# load data - only text
txt_docs = SimpleDirectoryReader(
    input_files=[pdf_path]
).load_data()


img_docs = SimpleDirectoryReader(input_dir=img_dir).load_data()

documents = txt_docs + img_docs

print("Total Text Documents:",len(txt_docs))
print("Total Image Documents:",len(img_docs))


# LanceDB Set Ups
text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)


index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)


retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=1
)


def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text



query_str = "Explain Scaled Dot-Product Attention."

img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)
img_doc = SimpleDirectoryReader(
    input_dir=img_dir, input_files=img
).load_data()
txt_doc = "\n".join(txt)




qa_tmpl_str = (
    "Given the provided information, including relevant images and retrieved context from the pdf file, \
 accurately and precisely answer the query without any additional prior knowledge.\n"
    "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
    "---------------------\n"
    "Context: {context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)


openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o", api_key=OPENAI_API_KEY, max_new_tokens=1500
)


response = openai_mm_llm.complete(
    prompt=qa_tmpl_str.format(
        context_str=txt_doc, query_str=query_str,
    ),
    image_documents=img_doc,
)


print("-"*50)
print("Answer:")
print(response.text)
