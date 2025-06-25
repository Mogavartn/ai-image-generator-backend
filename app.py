#!/usr/bin/env python3
"""
Générateur d'Images IA - Version allégée pour Render Free
Optimisé pour 512MB RAM
"""

import os
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import conditionnel des librairies lourdes
try:
    import safetensors.torch as st
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  Safetensors non disponible")

# Import diffusers seulement si nécessaire
DIFFUSERS_AVAILABLE = False

app = Flask(__name__)
CORS(app, origins=['https://webinterface-imageai.onrender.com'])

# Configuration ultra-légère
app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max
    UPLOAD_FOLDER=tempfile.mkdtemp(prefix="ai_lite_"),
    RESULTS_FOLDER="./generated_images",
    MAX_IMAGES=1,  # Une seule image max
    DEFAULT_STEPS=10,  # Steps très réduits
)

# Variables globales
temp_files = set()

class LightweightGenerator:
    """Générateur ultra-léger pour plan gratuit"""
    
    def __init__(self):
        self.device = "cpu"  # Forcé CPU pour le plan gratuit
        self.temp_dir = Path(app.config['UPLOAD_FOLDER'])
        self.results_dir = Path(app.config['RESULTS_FOLDER'])
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"🖥️  Device: {self.device}")
        print(f"📁 Mode ultra-léger activé")
    
    def validate_safetensors(self, file_path: Path) -> bool:
        """Validation minimale"""
        try:
            if not SAFETENSORS_AVAILABLE:
                # Validation basique par taille de fichier
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb < 0.1 or size_mb > 5000:  # Entre 100KB et 5GB
                    return False
                print(f"📊 Fichier: {size_mb:.1f}MB (validation basique)")
                return True
            
            # Validation complète si safetensors disponible
            st.load_file(str(file_path))
            return True
        except Exception as e:
            print(f"❌ Validation échouée: {e}")
            return False
    
    def generate_mock_images(self, 
                           prompt: str,
                           num_images: int = 1,
                           dimensions: str = "512x512") -> List[Dict]:
        """Génération d'images simulée pour les tests"""
        
        print(f"🎨 Génération simulée: {prompt}")
        
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i in range(min(num_images, 1)):  # Max 1 image en gratuit
            try:
                # Simuler un temps de génération
                time.sleep(2)  # 2 secondes de simulation
                
                # Créer une image de test (placeholder)
                from PIL import Image, ImageDraw, ImageFont
                
                width, height = map(int, dimensions.split('x'))
                
                # Créer une image simple
                img = Image.new('RGB', (width, height), color='lightblue')
                draw = ImageDraw.Draw(img)
                
                # Ajouter du texte
                text_lines = [
                    "Image générée",
                    f"Prompt: {prompt[:30]}...",
                    f"Dimensions: {dimensions}",
                    f"Mode: Simulation"
                ]
                
                y_offset = height // 4
                for line in text_lines:
                    try:
                        draw.text((10, y_offset), line, fill='black')
                        y_offset += 30
                    except:
                        pass  # Si police non disponible
                
                # Sauvegarder
                filename = f"generated_{timestamp}_{i+1:03d}.png"
                image_path = self.results_dir / filename
                img.save(str(image_path))
                
                results.append({
                    "filename": filename,
                    "path": str(image_path),
                    "dimensions": dimensions,
                    "generation_time": 2.0,
                    "url": f"/api/image/{filename}",
                    "mode": "simulation"
                })
                
                print(f"✅ Image simulée créée: {filename}")
                
            except Exception as e:
                print(f"❌ Erreur simulation: {e}")
                results.append({
                    "error": str(e),
                    "index": i+1
                })
        
        return results
    
    def generate_real_images(self, model_path: Path, **kwargs) -> List[Dict]:
        """Génération réelle (si ressources suffisantes)"""
        
        # Vérifier la mémoire disponible
        import psutil
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > 80:
            raise ValueError("Mémoire insuffisante pour la génération réelle")
        
        # Import dynamique de diffusers
        try:
            global DIFFUSERS_AVAILABLE
            if not DIFFUSERS_AVAILABLE:
                print("📥 Chargement de diffusers...")
                from diffusers import FluxPipeline
                DIFFUSERS_AVAILABLE = True
            
            # Charger le modèle (très lourd pour plan gratuit)
            print(f"🔄 Chargement modèle: {model_path}")
            pipeline = FluxPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32,  # CPU
                device_map=None
            )
            
            # Génération (très lente sur CPU)
            print("🎨 Génération en cours...")
            # ... logique de génération réelle
            
        except Exception as e:
            print(f"❌ Génération réelle impossible: {e}")
            raise ValueError(f"Génération réelle échouée: {e}")

# Instance globale
generator = LightweightGenerator()

@app.route('/api/generate', methods=['POST'])
def generate_images():
    """Endpoint allégé de génération"""
    try:
        print("📨 Nouvelle requête (mode léger)")
        
        # Validation basique
        if 'model_file' not in request.files:
            return jsonify({"success": False, "error": "Fichier modèle requis"}), 400
        
        model_file = request.files['model_file']
        if not model_file.filename:
            return jsonify({"success": False, "error": "Nom de fichier invalide"}), 400
        
        # Sauvegarder le fichier
        filename = secure_filename(model_file.filename)
        model_path = generator.temp_dir / f"model_{int(time.time())}_{filename}"
        model_file.save(str(model_path))
        temp_files.add(str(model_path))
        
        # Validation
        if not generator.validate_safetensors(model_path):
            return jsonify({"success": False, "error": "Fichier invalide"}), 400
        
        # Paramètres
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify({"success": False, "error": "Prompt requis"}), 400
        
        dimensions = request.form.get('dimensions', '512x512')
        
        # Mode de génération selon les ressources
        try:
            # Essayer génération réelle si possible
            results = generator.generate_real_images(
                model_path=model_path,
                prompt=prompt,
                dimensions=dimensions
            )
            mode = "real"
        except Exception as e:
            print(f"⚠️  Génération réelle impossible: {e}")
            # Fallback vers simulation
            results = generator.generate_mock_images(
                prompt=prompt,
                dimensions=dimensions
            )
            mode = "simulation"
        
        # Filtrer les succès
        successful_results = [r for r in results if 'error' not in r]
        
        return jsonify({
            "success": True,
            "images": successful_results,
            "total_generated": len(successful_results),
            "mode": mode,
            "message": "Génération en mode simulation (plan gratuit)" if mode == "simulation" else "Génération réelle"
        })
        
    except Exception as e:
        print(f"❌ Erreur serveur: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/image/<filename>')
def serve_image(filename):
    """Servir les images"""
    try:
        image_path = generator.results_dir / secure_filename(filename)
        if image_path.exists():
            return send_file(str(image_path))
        return jsonify({"error": "Image non trouvée"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_temp_files():
    """Nettoyage ultra-léger"""
    try:
        cleaned = 0
        for file_path in list(temp_files):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    cleaned += 1
                temp_files.discard(file_path)
            except:
                pass
        
        return jsonify({
            "success": True,
            "cleaned_files": cleaned,
            "message": f"{cleaned} fichiers nettoyés"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status')
def get_status():
    """Status ultra-léger"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        return jsonify({
            "status": "online",
            "device": generator.device,
            "mode": "lightweight",
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available // (1024*1024),
            "safetensors_available": SAFETENSORS_AVAILABLE,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "temp_files": len(temp_files)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/')
def index():
    """Page simple"""
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Générateur IA - Mode Léger</title></head>
    <body>
        <h1>🎨 Générateur d'Images IA - Mode Léger</h1>
        <p>Backend optimisé pour plan gratuit Render</p>
        <ul>
            <li><a href="/api/status">Status</a></li>
            <li>Mode: Simulation + génération réelle si possible</li>
            <li>Optimisé: 512MB RAM, CPU uniquement</li>
        </ul>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Démarrage mode ultra-léger")
    print(f"💾 Mémoire: Plan gratuit (512MB)")
    print(f"🔧 Safetensors: {'✅' if SAFETENSORS_AVAILABLE else '❌'}")
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)