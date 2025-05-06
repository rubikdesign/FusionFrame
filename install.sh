#!/bin/bash
# Script pentru crearea unui mediu virtual cu dependențe compatibile pentru ICEdit Pro

echo "===== Configurare mediu pentru ICEdit Pro ====="
echo "Creez mediul virtual..."

# Crearea mediului virtual
python -m venv icedit_env
source icedit_env/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "Instalez dependențele..."
pip install -r requirements.txt

echo "Creez script de execuție..."
# Creăm scriptul de execuție
cat > run_icedit.sh << 'EOL'
#!/bin/bash
source icedit_env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python app.py
EOL

chmod +x run_icedit.sh

echo "===== Instalare completă! ====="
echo "Pentru a rula aplicația, folosește comanda: ./run_icedit.sh"
echo "NOTĂ: La prima rulare, modelele vor fi descărcate automat (poate dura câteva minute)."