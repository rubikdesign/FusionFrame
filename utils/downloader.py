#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitar pentru descărcarea modelelor în FusionFrame 2.0
"""

import os
import requests
import logging
import zipfile
import gdown
import hashlib
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm

from config.app_config import AppConfig

# Setăm logger-ul
logger = logging.getLogger(__name__)

class ModelDownloader:
    """
    Utilitar pentru descărcarea și verificarea modelelor
    
    Responsabil pentru descărcarea modelelor necesare din surse diverse
    (Hugging Face, Google Drive, etc) și verificarea integrității acestora.
    """
    
    def __init__(self):
        """Inițializează downloader-ul pentru modele"""
        self.config = AppConfig
    
    def download_file(self, url: str, destination: str, 
                     expected_hash: Optional[str] = None) -> bool:
        """
        Descarcă un fișier de la URL către destinație
        
        Args:
            url: URL-ul fișierului de descărcat
            destination: Calea unde va fi salvat fișierul
            expected_hash: Valoarea hash așteptată pentru verificare (opțional)
            
        Returns:
            True dacă descărcarea a reușit, False altfel
        """
        try:
            # Creăm directorul destinație dacă nu există
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Descărcăm fișierul cu progres
            logger.info(f"Downloading file from {url} to {destination}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=f"Downloading {os.path.basename(destination)}") as pbar:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verificăm hash-ul dacă este furnizat
            if expected_hash:
                if not self.verify_file_hash(destination, expected_hash):
                    logger.error(f"Hash verification failed for {destination}")
                    return False
            
            logger.info(f"Download complete: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False
    
    def download_from_gdrive(self, file_id: str, destination: str, 
                           expected_hash: Optional[str] = None) -> bool:
        """
        Descarcă un fișier de pe Google Drive
        
        Args:
            file_id: ID-ul fișierului Google Drive
            destination: Calea unde va fi salvat fișierul
            expected_hash: Valoarea hash așteptată pentru verificare (opțional)
            
        Returns:
            True dacă descărcarea a reușit, False altfel
        """
        try:
            # Creăm directorul destinație dacă nu există
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Descărcăm fișierul
            logger.info(f"Downloading file from Google Drive (ID: {file_id}) to {destination}")
            gdown.download(id=file_id, output=destination, quiet=False)
            
            # Verificăm hash-ul dacă este furnizat
            if expected_hash:
                if not self.verify_file_hash(destination, expected_hash):
                    logger.error(f"Hash verification failed for {destination}")
                    return False
            
            logger.info(f"Download complete: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from Google Drive: {str(e)}")
            return False
    
    def download_and_extract(self, url: str, destination_dir: str, 
                            expected_hash: Optional[str] = None) -> bool:
        """
        Descarcă și extrage un fișier zip
        
        Args:
            url: URL-ul fișierului zip
            destination_dir: Directorul unde va fi extras fișierul
            expected_hash: Valoarea hash așteptată pentru verificare (opțional)
            
        Returns:
            True dacă descărcarea și extragerea au reușit, False altfel
        """
        try:
            # Creăm directorul destinație dacă nu există
            os.makedirs(destination_dir, exist_ok=True)
            
            # Fișier temporar pentru zip
            temp_zip = os.path.join(destination_dir, "temp.zip")
            
            # Descărcăm fișierul zip
            if not self.download_file(url, temp_zip, expected_hash):
                return False
            
            # Extragem fișierul zip
            logger.info(f"Extracting zip file to {destination_dir}")
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(destination_dir)
            
            # Ștergem fișierul temporar zip
            os.remove(temp_zip)
            
            logger.info(f"Extraction complete to {destination_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting zip file: {str(e)}")
            return False
    
    def verify_file_hash(self, file_path: str, expected_hash: str) -> bool:
        """
        Verifică integritatea fișierului prin hash
        
        Args:
            file_path: Calea către fișier
            expected_hash: Valoarea hash așteptată
            
        Returns:
            True dacă hash-ul este corect, False altfel
        """
        try:
            # Calculăm hash-ul fișierului
            logger.info(f"Verifying hash for {file_path}")
            hash_md5 = hashlib.md5()
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            file_hash = hash_md5.hexdigest()
            
            # Verificăm hash-ul
            if file_hash.lower() == expected_hash.lower():
                logger.info(f"Hash verification successful for {file_path}")
                return True
            else:
                logger.error(f"Hash mismatch for {file_path}")
                logger.error(f"  Expected: {expected_hash}")
                logger.error(f"  Got: {file_hash}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying file hash: {str(e)}")
            return False
    
    def ensure_model(self, model_name: str, model_url: str, expected_hash: Optional[str] = None) -> str:
        """
        Se asigură că un model specific este descărcat și disponibil
        
        Args:
            model_name: Numele modelului
            model_url: URL-ul pentru descărcare
            expected_hash: Valoarea hash așteptată pentru verificare (opțional)
            
        Returns:
            Calea către modelul descărcat sau None în caz de eroare
        """
        model_path = os.path.join(self.config.MODEL_DIR, model_name)
        
        # Verificăm dacă modelul există
        if os.path.exists(model_path):
            # Dacă există și avem un hash, verificăm integritatea
            if expected_hash and not self.verify_file_hash(model_path, expected_hash):
                logger.warning(f"Hash verification failed for existing model {model_name}. Re-downloading...")
            else:
                logger.info(f"Model {model_name} already exists")
                return model_path
        
        # Descompunem URL-ul pentru a determina sursa
        if "drive.google.com" in model_url or "https://drive.google.com" in model_url:
            # Extragem ID-ul Google Drive
            file_id = model_url.split("/")[-2]
            if self.download_from_gdrive(file_id, model_path, expected_hash):
                return model_path
        elif model_url.endswith(".zip"):
            # Descărcăm și extragem zip-ul
            if self.download_and_extract(model_url, os.path.dirname(model_path), expected_hash):
                return model_path
        else:
            # Descărcare simplă
            if self.download_file(model_url, model_path, expected_hash):
                return model_path
        
        return None