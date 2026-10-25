# 问答系统总架构

源文件 ──Extractor──▶ Document
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   AnswerService（读）              ArchiveService（写）
   当前文档原文                     切片 → 嵌入 → upsert
   + 检索历史知识库 top-k                  │
          │                               ▼
          ▼                          VectorStore
       LLM 回答                     （持续累积）
## 两条分支的关系
1 执行时序

并行执行，不要互相阻塞

def process(self, source: str) -> Document:
    document = self.extractor.extract(source)

    # 写分支：后台跑，失败只记日志
    #用户在存档区域实现文件导入，切片存档，也就是utils里的pipeline流程，当然也要支持批量导入。
    if self.config.archive.enabled:#这里指的是存档区域，一种状态激活，可以配合前端的（上传文件）开关触发
        self._submit_background(self._archive_safe, document)

    # 读分支：正常返回，不等存档
    return document

def ask(self, document: Document, question: str) -> str:
    # 检索时排除当前文档自身切片（见层面二）
    history = self.retriever.search(
        question,
        collection=self.config.archive.collection,
        exclude_document_id=document.id,
    )
    return self.answer_service.answer(document, question, history)

2.数据可见性——回答时不能看到"刚存进去的自己"，必须按documentid排除自身切片
class Retriever:
    def search(self, query, collection, exclude_document_id=None, top_k=5):
        results = self.store.query(collection, query, top_k=top_k * 2)
        if exclude_document_id:
            results = [
                r for r in results
                if r.metadata.get("document_id") != exclude_document_id
            ]
        return results[:top_k]

3.产物回流,针对当前文件的回答情况，也要记录到历史知识库中

该分支是独立于文档存档，应该是问答界面有一个按钮，控制将选定的对话内容进行存档，当然最好添加一个用户修正按钮，也就是在问答内容保存前，有一个弹框，用户可以确认是否修正，修正后再保存。
@dataclass
class ArchiveConfig:
    enabled: bool = False              # 文档切片存档
    collection: str = "default"
    chunk_size: int = 512
    chunk_overlap: int = 64
    # ↓ 新增：回答产物回流
    persist_qa: bool = False           # 问答对是否写回
    qa_collection: str = "qa_pairs"

#修订的问答服务
def ask(self, document: Document, question: str) -> str:
    history = self.retriever.search(question, ..., exclude_document_id=document.id)
    answer = self.answer_service.answer(document, question, history)

    if self.config.archive.persist_qa:#这里就是控制问答回流的地方
        #然后增加修正弹框的控制，用户确认是否修正，修正后再保存
        if xxx:
            #用户确认修正，再保存
            answer = process(answer)
        self._archive_qa_safe(document, question, answer)   # 同样后台、同样吞异常
    return answer


def _archive_qa_safe(self, document, question, answer):
    try:
        text = f"Q: {question}\nA: {answer}"
        self.store.upsert(
            collection=self.config.archive.qa_collection,
            records=[{
                "id": f"qa:{document.id}:{hash(question)}",
                "vector": self.embedder.embed([text])[0],
                "text": text,
                "metadata": {
                    "type": "qa",
                    "source_document_id": document.id,
                    "source": document.source,
                },
            }],
        )
    except Exception:
        log.exception("persist_qa failed, ignored")
