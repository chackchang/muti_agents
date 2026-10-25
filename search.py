import asyncio
from utils.db.db_utils import initialize_database, close_database, vector_search, hybrid_search
from utils.embedding.embedding_runtime import aembed_texts

async def main():
    await initialize_database()
    query = " What are the novel algorithms used for laser centerline extraction and spatial curve fitting in the WA-DED inspection system?"
    (dense, sparse) = (await aembed_texts([query]))[0]

    # print("[1] 纯稠密 vector_search:")
    # for r in await vector_search(dense, limit=5):
    #     print(f"  {r['similarity']:.4f} {r['document_title']} {r['content']}")

    print("[2] 混合 hybrid_search (RRF):")
    for r in await hybrid_search(dense, query_text=query, limit=5, sparse=sparse):
        print(f"  {r['combined_score']:.4f} {r['document_title']} {r['content']}")

    await close_database()

asyncio.run(main())