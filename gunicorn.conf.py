# gunicorn.conf.py - Configuration optimisée pour Render Free
import os

# Configuration pour plan gratuit Render (0.1 CPU, 512MB RAM)
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"

# Un seul worker pour économiser la RAM
workers = 1
worker_class = "sync"
worker_connections = 10

# Timeouts adaptés au plan gratuit
timeout = 600  # 10 minutes pour génération d'images
keepalive = 2
max_requests = 50  # Redémarrer le worker après 50 requêtes
max_requests_jitter = 10

# Gestion mémoire
worker_memory_limit = 400 * 1024 * 1024  # 400MB limite par worker
preload_app = True  # Précharger l'app pour économiser RAM

# Logs
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Optimisations pour démarrage
pythonpath = "/opt/render/project/src"

def when_ready(server):
    """Callback exécuté quand le serveur est prêt"""
    print("🚀 Serveur Gunicorn prêt - Plan Free Render")
    print(f"💾 RAM limite: {worker_memory_limit // (1024*1024)}MB par worker")

def worker_exit(server, worker):
    """Cleanup quand un worker se ferme"""
    print(f"🔄 Worker {worker.pid} fermé - libération mémoire")
    
    # Nettoyage mémoire PyTorch
    try:
        import torch
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except:
        pass