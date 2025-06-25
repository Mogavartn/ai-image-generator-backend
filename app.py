#!/usr/bin/env python3
"""
Générateur d'Images IA - Backend Python
Support pour modèles Flux et Hunyuan avec LoRAs
"""

import os
import json
import time
import hashlib
import requests
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import safetensors.torch as st

# Imports spécifiques aux modèles (à adapter selon vos librairies)
try:
    from diffusers import FluxPipeline, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  Diffusers non installé. Installez avec: pip install diffusers")

try:
    # Import hypothétique pour Hunyuan (remplacez par la vraie librairie)
    from hunyuan_dit import HunyuanPipeline
    HUNYUAN_AVAILABLE = True
except ImportError:
    HUNYUAN_AVAILABLE = False
    print("⚠️  Librairie Hunyuan non trouvée. Veuillez l'installer.")

app = Flask(__name__)
CORS(app, origins=['https://webinterface-imageai.onrender.com'])

# Configuration
app.config.update(
    MAX_CONTENT_LENGTH=2 * 1024 * 1024 * 1024,  # 2GB max
    UPLOAD_FOLDER=tempfile.mkdtemp(prefix="ai_gen_"),
    RESULTS_FOLDER="./generated_images",
    ALLOWED_EXTENSIONS={'.safetensors'},
    MAX_IMAGES=4,
    DEFAULT_STEPS=30,
    DEFAULT_GUIDANCE=7.5
)

# Variables globales pour les pipelines chargés
loaded_pipelines = {}
temp_files = set()

class ImageGenerator:
    """Gestionnaire principal pour la génération d'images"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = Path(app.config['UPLOAD_FOLDER'])
        self.results_dir = Path(app.config['RESULTS_FOLDER'])
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"🖥️  Device: {self.device}")
        print(f"📁 Dossier temporaire: {self.temp_dir}")
        print(f"📁 Dossier résultats: {self.results_dir}")
    
    def validate_safetensors(self, file_path: Path) -> bool:
        """Valide qu'un fichier safetensors est lisible"""
        try:
            st.load_file(str(file_path))
            return True
        except Exception as e:
            print(f"❌ Erreur validation safetensors: {e}")
            return False
    
    def download_lora(self, url: str) -> Optional[Path]:
        """Télécharge un LoRA depuis une URL"""
        try:
            print(f"📥 Téléchargement LoRA: {url}")
            
            # Générer un nom de fichier sécurisé
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or "lora.safetensors"
            if not filename.endswith('.safetensors'):
                filename += '.safetensors'
            
            local_path = self.temp_dir / f"lora_{hashlib.md5(url.encode()).hexdigest()[:8]}_{filename}"
            
            # Télécharger avec timeout
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            # Valider le fichier téléchargé
            if self.validate_safetensors(local_path):
                temp_files.add(str(local_path))
                print(f"✅ LoRA téléchargé: {local_path}")
                return local_path
            else:
                local_path.unlink(missing_ok=True)
                print(f"❌ LoRA invalide: {url}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur téléchargement LoRA {url}: {e}")
            return None
    
    def load_flux_pipeline(self, model_path: Path, lora_paths: List[Path]) -> Optional[object]:
        """Charge un pipeline Flux avec LoRAs"""
        if not DIFFUSERS_AVAILABLE:
            raise ValueError("Diffusers non disponible")
        
        try:
            print(f"🔄 Chargement modèle Flux: {model_path}")
            
            # Charger le pipeline de base
            pipeline = FluxPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Charger les LoRAs
            for lora_path in lora_paths:
                print(f"🔗 Ajout LoRA: {lora_path}")
                pipeline.load_lora_weights(str(lora_path))
            
            pipeline = pipeline.to(self.device)
            return pipeline
            
        except Exception as e:
            print(f"❌ Erreur chargement Flux: {e}")
            return None
    
    def load_hunyuan_pipeline(self, model_path: Path, lora_paths: List[Path]) -> Optional[object]:
        """Charge un pipeline Hunyuan avec LoRAs"""
        if not HUNYUAN_AVAILABLE:
            raise ValueError("Librairie Hunyuan non disponible")
        
        try:
            print(f"🔄 Chargement modèle Hunyuan: {model_path}")
            
            # Adaptation nécessaire selon l'API Hunyuan réelle
            pipeline = HunyuanPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Charger les LoRAs (API à adapter)
            for lora_path in lora_paths:
                print(f"🔗 Ajout LoRA Hunyuan: {lora_path}")
                # pipeline.load_lora_weights(str(lora_path))  # À adapter
            
            pipeline = pipeline.to(self.device)
            return pipeline
            
        except Exception as e:
            print(f"❌ Erreur chargement Hunyuan: {e}")
            return None
    
    def parse_dimensions(self, dimensions_str: str) -> Tuple[int, int]:
        """Parse les dimensions au format 'WxH'"""
        try:
            width, height = map(int, dimensions_str.split('x'))
            return width, height
        except:
            return 512, 512
    
    def generate_images(self, 
                       model_path: Path,
                       model_type: str,
                       prompt: str,
                       lora_paths: List[Path] = None,
                       num_images: int = 1,
                       dimensions: str = "512x512",
                       steps: int = 30,
                       guidance_scale: float = 7.5,
                       output_dir: str = None) -> List[Dict]:
        """Génère les images selon les paramètres"""
        
        if lora_paths is None:
            lora_paths = []
        
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer une clé unique pour le pipeline
        pipeline_key = f"{model_type}_{model_path.stem}_{len(lora_paths)}"
        
        # Charger ou récupérer le pipeline
        if pipeline_key not in loaded_pipelines:
            if model_type.lower() == "flux":
                pipeline = self.load_flux_pipeline(model_path, lora_paths)
            elif model_type.lower() == "hunyuan":
                pipeline = self.load_hunyuan_pipeline(model_path, lora_paths)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
            
            if pipeline is None:
                raise ValueError(f"Impossible de charger le pipeline {model_type}")
            
            loaded_pipelines[pipeline_key] = pipeline
        else:
            pipeline = loaded_pipelines[pipeline_key]
            print(f"♻️  Réutilisation pipeline: {pipeline_key}")
        
        # Parser les dimensions
        width, height = self.parse_dimensions(dimensions)
        
        # Générer les images
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i in range(num_images):
            try:
                print(f"🎨 Génération image {i+1}/{num_images}")
                start_time = time.time()
                
                # Générer l'image
                with torch.no_grad():
                    image = pipeline(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(
                            int(time.time()) + i
                        )
                    ).images[0]
                
                generation_time = time.time() - start_time
                
                # Sauvegarder l'image
                filename = f"generated_{timestamp}_{i+1:03d}.png"
                image_path = output_dir / filename
                image.save(str(image_path))
                
                results.append({
                    "filename": filename,
                    "path": str(image_path),
                    "dimensions": f"{width}x{height}",
                    "generation_time": round(generation_time, 2),
                    "url": f"/api/image/{filename}"
                })
                
                print(f"✅ Image {i+1} générée en {generation_time:.2f}s: {filename}")
                
            except Exception as e:
                print(f"❌ Erreur génération image {i+1}: {e}")
                results.append({
                    "error": str(e),
                    "index": i+1
                })
        
        return results

# Instance globale du générateur
generator = ImageGenerator()

@app.route('/api/generate', methods=['POST'])
def generate_images():
    """Endpoint principal pour générer des images"""
    try:
        print("📨 Nouvelle requête de génération")
        
        # Validation des données reçues
        if 'model_file' not in request.files:
            return jsonify({"success": False, "error": "Aucun fichier modèle fourni"}), 400
        
        model_file = request.files['model_file']
        if model_file.filename == '':
            return jsonify({"success": False, "error": "Nom de fichier invalide"}), 400
        
        # Sauvegarder le fichier modèle
        filename = secure_filename(model_file.filename)
        if not filename.endswith('.safetensors'):
            return jsonify({"success": False, "error": "Format de fichier non supporté"}), 400
        
        model_path = generator.temp_dir / f"model_{int(time.time())}_{filename}"
        model_file.save(str(model_path))
        temp_files.add(str(model_path))
        
        # Valider le modèle
        if not generator.validate_safetensors(model_path):
            return jsonify({"success": False, "error": "Fichier safetensors invalide"}), 400
        
        # Récupérer les paramètres
        model_type = request.form.get('model_type', 'flux')
        prompt = request.form.get('prompt', '').strip()
        lora_urls = request.form.get('lora_urls', '').strip()
        output_dir = request.form.get('output_dir', str(generator.results_dir))
        image_count = min(int(request.form.get('image_count', 1)), app.config['MAX_IMAGES'])
        dimensions = request.form.get('dimensions', '512x512')
        steps = int(request.form.get('steps', app.config['DEFAULT_STEPS']))
        guidance = float(request.form.get('guidance', app.config['DEFAULT_GUIDANCE']))
        
        if not prompt:
            return jsonify({"success": False, "error": "Prompt requis"}), 400
        
        print(f"🎯 Paramètres: {model_type}, {image_count} images, {dimensions}")
        
        # Télécharger les LoRAs
        lora_paths = []
        if lora_urls:
            urls = [url.strip() for url in lora_urls.split('\n') if url.strip()]
            for url in urls:
                lora_path = generator.download_lora(url)
                if lora_path:
                    lora_paths.append(lora_path)
        
        # Générer les images
        results = generator.generate_images(
            model_path=model_path,
            model_type=model_type,
            prompt=prompt,
            lora_paths=lora_paths,
            num_images=image_count,
            dimensions=dimensions,
            steps=steps,
            guidance_scale=guidance,
            output_dir=output_dir
        )
        
        # Filtrer les résultats avec erreurs
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        response = {
            "success": True,
            "images": successful_results,
            "total_generated": len(successful_results),
            "total_requested": image_count
        }
        
        if failed_results:
            response["errors"] = failed_results
        
        print(f"✅ Génération terminée: {len(successful_results)}/{image_count} images")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erreur serveur: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/image/<filename>')
def serve_image(filename):
    """Servir les images générées"""
    try:
        image_path = generator.results_dir / secure_filename(filename)
        if image_path.exists():
            return send_file(str(image_path))
        else:
            return jsonify({"error": "Image non trouvée"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_temp_files():
    """Nettoyer les fichiers temporaires"""
    try:
        cleaned_count = 0
        
        # Nettoyer les fichiers temporaires trackés
        for file_path in list(temp_files):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    cleaned_count += 1
                temp_files.discard(file_path)
            except Exception as e:
                print(f"⚠️  Erreur suppression {file_path}: {e}")
        
        # Nettoyer le dossier temporaire
        try:
            for item in generator.temp_dir.iterdir():
                if item.is_file():
                    item.unlink()
                    cleaned_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    cleaned_count += 1
        except Exception as e:
            print(f"⚠️  Erreur nettoyage dossier temp: {e}")
        
        # Vider le cache des pipelines si demandé
        global loaded_pipelines
        if request.json and request.json.get('clear_models', False):
            loaded_pipelines.clear()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("🧹 Cache modèles vidé")
        
        print(f"🧹 Nettoyage terminé: {cleaned_count} éléments supprimés")
        return jsonify({
            "success": True,
            "cleaned_files": cleaned_count,
            "message": f"{cleaned_count} fichiers temporaires supprimés"
        })
        
    except Exception as e:
        print(f"❌ Erreur nettoyage: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status')
def get_status():
    """Status de l'application"""
    return jsonify({
        "status": "online",
        "device": generator.device,
        "loaded_models": len(loaded_pipelines),
        "temp_files": len(temp_files),
        "diffusers_available": DIFFUSERS_AVAILABLE,
        "hunyuan_available": HUNYUAN_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "memory_used": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    })

@app.route('/')
def index():
    """Page d'accueil - servir l'interface web"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Générateur d'Images IA</title>
    </head>
    <body>
        <h1>🎨 Générateur d'Images IA - Backend</h1>
        <p>Le backend Python est en fonctionnement!</p>
        <ul>
            <li><a href="/api/status">Status de l'API</a></li>
            <li><strong>POST /api/generate</strong> - Générer des images</li>
            <li><strong>POST /api/cleanup</strong> - Nettoyer les fichiers temporaires</li>
            <li><strong>GET /api/image/&lt;filename&gt;</strong> - Servir les images</li>
        </ul>
        <p>Utilisez l'interface web HTML pour interagir avec cette API.</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Démarrage du serveur de génération d'images IA")
    print(f"📊 Device: {generator.device}")
    print(f"🔧 Diffusers: {'✅' if DIFFUSERS_AVAILABLE else '❌'}")
    print(f"🔧 Hunyuan: {'✅' if HUNYUAN_AVAILABLE else '❌'}")
    
    # Configuration pour le développement
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0' if not debug_mode else '127.0.0.1')
    
    app.run(
        host=host,
        port=port,
        debug=debug_mode,
        threaded=True
    )