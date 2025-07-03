from celery import Celery

app = Celery('llm_tasks', backend='redis://localhost:6379', broker='redis://localhost:6379')
