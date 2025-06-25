#!/bin/bash
echo "🏠 Configuration de l'environnement local..."

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé. Veuillez l'installer."
    exit 1
fi

# Créer un environnement virtuel
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🔄 Activation de l'environnement virtuel..."
source venv/bin/activate

# Mettre à jour pip
echo "📥 Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances de base
echo "📦 Installation des dépendances de base..."
pip install -r requirements.txt

# Détecter CUDA et installer PyTorch approprié
echo "🔍 Détection CUDA..."
if python3 -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>/dev/null; then
    echo "🔥 Installation de PyTorch (version détectée automatiquement)"
else
    echo "💻 Installation de PyTorch CPU"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Créer la structure de dossiers
echo "📁 Création des dossiers..."
mkdir -p generated_images temp models

# Créer un fichier .env
cat > .env << 'ENVEOF'
FLASK_DEBUG=true
FLASK_ENV=development
HOST=127.0.0.1
PORT=5000
ENVEOF

echo "✅ Configuration terminée!"
echo ""
echo "🚀 Pour démarrer l'application:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "🌐 L'application sera disponible sur http://localhost:5000"
