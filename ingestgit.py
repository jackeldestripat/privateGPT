#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import  GitLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from constants import CHROMA_SETTINGS
import argparse

def main():
    # Parse the command line arguments
    args = parse_arguments()
    print(args.url)
    print(args.name)
    load(args.url, args.name)


load_dotenv()
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50


def load(url: str, name: str):
    repo_path = "source_git/"+name

    if not os.path.exists(repo_path):
        loader = GitLoader(repo_path, clone_url=url)
        documents = loader.load()

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(total=len(documents), desc='Loading new documents', ncols=80) as pbar:
                for document in documents:
                    results.extend(document)
                    pbar.update()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = text_splitter.split_documents(documents)

            print(f"Appending to existing vectorstore at {persist_directory}")
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
            collection = db.get()
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)

        return results
    else:
        print("repo already exist")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Ingest a remote git repo')
    parser.add_argument("--url", "-u", type=str, required=True,
                        help='URL of the Git repository to ingest.')

    parser.add_argument("--name", "-n", type=str, required=True,
                        help='Name of the local repository.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
