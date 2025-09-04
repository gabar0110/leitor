import re
import sqlite3
from pathlib import Path
import PyPDF2
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from heapq import nlargest
from collections import defaultdict
import unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import requests
import json
import io

# Carrega o modelo de linguagem natural
try:
    nlp = spacy.load("pt_core_news_sm")
except:
    print("Instale o modelo de linguagem: python -m spacy download pt_core_news_sm")
    exit()

class GerenciadorBancoDados:
    def __init__(self, db_name='recrutamento.db'):
        self.conn = sqlite3.connect(db_name)
        self.criar_tabelas()
    
    def criar_tabelas(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidatos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            caminho_curriculo TEXT,
            resumo TEXT,
            palavras_chave TEXT,
            data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            area_interesse TEXT,
            nivel_experiencia TEXT
        )''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS classificacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidato_id INTEGER,
            area_interesse TEXT,
            nivel_experiencia TEXT,
            data_classificacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (candidato_id) REFERENCES candidatos (id)
        )''')
        self.conn.commit()
    
    def inserir_candidato(self, nome, caminho, resumo, palavras_chave, area_interesse=None, nivel_experiencia=None):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO candidatos (nome, caminho_curriculo, resumo, palavras_chave, area_interesse, nivel_experiencia)
        VALUES (?, ?, ?, ?, ?, ?)''', (nome, caminho, resumo, ','.join(palavras_chave), area_interesse, nivel_experiencia))
        self.conn.commit()
        return cursor.lastrowid
    
    def atualizar_classificacao(self, candidato_id, area_interesse, nivel_experiencia):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO classificacoes (candidato_id, area_interesse, nivel_experiencia)
        VALUES (?, ?, ?)''', (candidato_id, area_interesse, nivel_experiencia))
        
        cursor.execute('''
        UPDATE candidatos 
        SET area_interesse = ?, nivel_experiencia = ?
        WHERE id = ?''', (area_interesse, nivel_experiencia, candidato_id))
        self.conn.commit()
    
    def buscar_todos(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, nome, caminho_curriculo, resumo, palavras_chave, area_interesse, nivel_experiencia FROM candidatos')
        return cursor.fetchall()
    
    def buscar_historico_classificacao(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT c.nome, cl.area_interesse, cl.nivel_experiencia, cl.data_classificacao
        FROM classificacoes cl
        JOIN candidatos c ON cl.candidato_id = c.id
        ORDER BY cl.data_classificacao DESC
        ''')
        return cursor.fetchall()
    
    def fechar_conexao(self):
        self.conn.close()

class ProcessadorCurriculos:
    @staticmethod
    def normalizar_texto(texto):
        """Remove acentos e caracteres especiais"""
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
        return texto.lower()
    
    @staticmethod
    def ler_arquivo(caminho):
        try:
            extensao = Path(caminho).suffix.lower()
            if extensao == '.txt':
                with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif extensao == '.pdf':
                texto = ''
                with open(caminho, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for pagina in pdf_reader.pages:
                        texto += pagina.extract_text() + '\n'
                return texto
            elif extensao == '.docx':
                doc = docx.Document(caminho)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Erro ao ler arquivo {caminho}: {e}")
            return None
    
    @staticmethod
    def extrair_nome(texto):
        if not texto:
            return "Candidato Desconhecido"
            
        # Primeiro tenta encontrar padrões comuns em currículos
        padroes_nome = [
            r"(?i)nome[\s]*:[\s]*([^\n]+)",
            r"(?i)name[\s]*:[\s]*([^\n]+)",
            r"(?i)candidato[\s]*:[\s]*([^\n]+)",
            r"^([A-ZÀ-Ü][a-zà-ü]+(?:\s+[A-ZÀ-Ü][a-zà-ü]+)+)",
            r"\b([A-ZÀ-Ü][a-zà-ü]+\s+[A-ZÀ-Ü][a-zà-ü]+)\b"
        ]
        
        for padrao in padroes_nome:
            match = re.search(padrao, texto)
            if match:
                nome = match.group(1).strip()
                if len(nome.split()) >= 2:  # Pelo menos nome e sobrenome
                    return nome.title()
        
        # Se não encontrar, usa o modelo de NLP
        try:
            doc = nlp(texto[:1000])  # Analisa apenas o início para performance
            for ent in doc.ents:
                if ent.label_ == "PER" and len(ent.text.split()) >= 2:
                    return ent.text.title()
        except:
            pass
        
        # Último recurso: usa o nome do arquivo
        return Path(texto.split('\n')[0]).stem.replace('_', ' ').title()
    
    @staticmethod
    def extrair_contato(texto):
        if not texto:
            return "Email: Não encontrado | Telefone: Não encontrado"
            
        # Extrai email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', texto)
        email = email_match.group(0) if email_match else "Não encontrado"
        
        # Extrai telefone (formatos brasileiros)
        telefone_patterns = [
            r'\(?\d{2}\)?[\s-]?\d{4,5}[\s-]?\d{4}',
            r'\d{2}[\s\.-]?\d{4,5}[\s\.-]?\d{4}',
            r'\+?\d{2}[\s-]?\(?\d{2}\)?[\s-]?\d{4,5}[\s-]?\d{4}'
        ]
        
        telefone = "Não encontrado"
        for pattern in telefone_patterns:
            match = re.search(pattern, texto)
            if match:
                telefone = match.group(0)
                break
        
        return f"Email: {email} | Telefone: {telefone}"
    
    @staticmethod
    def lematizar_texto(texto):
        """Lematização aprimorada do texto"""
        if not texto:
            return ""
            
        doc = nlp(texto)
        lemas = []
        
        for token in doc:
            # Ignora stopwords, pontuação e espaços
            if not token.is_stop and not token.is_punct and not token.is_space:
                # Usa o lemma se disponível, caso contrário, usa o texto original
                lemma = token.lemma_.lower().strip()
                if lemma and len(lemma) > 1:  # Ignora lemas muito curtos
                    lemas.append(lemma)
        
        return " ".join(lemas)
    
    @staticmethod
    def processar_texto(texto):
        if not texto:
            return []
            
        doc = nlp(texto)
        
        # Extrair palavras-chave (nomes, habilidades, tecnologias)
        palavras_chave = set()
        
        # Padrões para habilidades técnicas
        padroes_tecnicos = [
            r'(?i)\b(python|java|c\+\+|c#|javascript|typescript|ruby|php|swift|kotlin|go|rust|scala|r)\b',
            r'(?i)\b(sql|mysql|postgresql|mongodb|oracle|sqlite|nosql)\b',
            r'(?i)\b(html|css|react|angular|vue|node\.?js|django|flask|spring|express)\b',
            r'(?i)\b(machine\slearning|deep\slearning|ai|artificial intelligence|nlp|computer\svision)\b',
            r'(?i)\b(aws|azure|google\scloud|docker|kubernetes|ci/cd|devops)\b',
            r'(?i)\b(project\smanagement|agile|scrum|kanban|devops|git|jenkins)\b'
        ]
        
        for padrao in padroes_tecnicos:
            for match in re.finditer(padrao, texto):
                palavras_chave.add(match.group(1).lower())
        
        # Extrair entidades nomeadas (organizações, tecnologias)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "MISC"]:
                palavras_chave.add(ent.text.lower())
        
        # Extrair substantivos, adjetivos e verbos relevantes com lematização
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                # Usa o lema em vez do texto original
                palavras_chave.add(token.lemma_.lower())
        
        # Extrair bigramas (combinações de palavras) com lematização
        for i in range(len(doc) - 1):
            if (doc[i].pos_ in ['NOUN', 'ADJ'] and 
                doc[i+1].pos_ == 'NOUN' and 
                not doc[i].is_stop and 
                not doc[i+1].is_stop):
                bigrama = f"{doc[i].lemma_} {doc[i+1].lemma_}".lower()
                palavras_chave.add(bigrama)
        
        return list(palavras_chave)
    
    @staticmethod
    def resumir_texto(texto, percentual=0.3):
        if not texto:
            return "Texto não disponível para resumo."
            
        try:
            doc = nlp(texto)
            frequencia = defaultdict(int)
            
            for token in doc:
                if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    # Usa o lema para calcular frequência
                    lemma = token.lemma_.lower()
                    frequencia[lemma] += 1
            
            if not frequencia:
                return texto[:500] + "..." if len(texto) > 500 else texto
                
            max_freq = max(frequencia.values())
            for palavra in frequencia:
                frequencia[palavra] /= max_freq
            
            pontuacao = {}
            for i, sent in enumerate(doc.sents):
                for token in sent:
                    lemma = token.lemma_.lower()
                    if lemma in frequencia:
                        pontuacao[i] = pontuacao.get(i, 0) + frequencia[lemma]
            
            if not pontuacao:
                return texto[:500] + "..." if len(texto) > 500 else texto
                
            qtd = max(1, int(len(pontuacao) * percentual))
            melhores = nlargest(qtd, pontuacao, key=pontuacao.get)
            melhores.sort()
            
            resumo = ' '.join(str(list(doc.sents)[i]) for i in melhores)
            
            # Adiciona informações de contato ao resumo
            contato = ProcessadorCurriculos.extrair_contato(texto)
            return f"{resumo}\n\nInformações de Contato:\n{contato}"
        except Exception as e:
            print(f"Erro ao gerar resumo: {e}")
            return texto[:500] + "..." if len(texto) > 500 else texto

class ClassificadorML:
    def __init__(self):
        self.modelo_area = None
        self.modelo_nivel = None
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.le_area = LabelEncoder()
        self.le_nivel = LabelEncoder()
        self.carregar_modelos()
    
    def carregar_modelos(self):
        try:
            self.modelo_area = joblib.load('modelo_area.pkl')
            self.modelo_nivel = joblib.load('modelo_nivel.pkl')
            self.vectorizer = joblib.load('vectorizer.pkl')
            self.le_area = joblib.load('le_area.pkl')
            self.le_nivel = joblib.load('le_nivel.pkl')
            return True
        except:
            # Modelos não existem, serão treinados quando houver dados suficientes
            return False
    
    def salvar_modelos(self):
        if self.modelo_area:
            joblib.dump(self.modelo_area, 'modelo_area.pkl')
            joblib.dump(self.modelo_nivel, 'modelo_nivel.pkl')
            joblib.dump(self.vectorizer, 'vectorizer.pkl')
            joblib.dump(self.le_area, 'le_area.pkl')
            joblib.dump(self.le_nivel, 'le_nivel.pkl')
    
    def treinar_modelos(self, textos, areas, niveis):
        if len(set(areas)) < 2 or len(set(niveis)) < 2:
            return False  # Não há dados suficientes para treinar
        
        # Vetoriza os textos
        X = self.vectorizer.fit_transform(textos)
        
        # Codifica as labels
        y_area = self.le_area.fit_transform(areas)
        y_nivel = self.le_nivel.fit_transform(niveis)
        
        # Treina o modelo para área
        self.modelo_area = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.modelo_area.fit(X, y_area)
        
        # Treina o modelo para nível de experiência
        self.modelo_nivel = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.modelo_nivel.fit(X, y_nivel)
        
        self.salvar_modelos()
        return True
    
    def prever(self, texto):
        if not self.modelo_area or not self.modelo_nivel:
            return "Desconhecida", "Desconhecido"
        
        # Vetoriza o texto
        X = self.vectorizer.transform([texto])
        
        # Faz as previsões
        area_idx = self.modelo_area.predict(X)[0]
        nivel_idx = self.modelo_nivel.predict(X)[0]
        
        # Decodifica as previsões
        area = self.le_area.inverse_transform([area_idx])[0]
        nivel = self.le_nivel.inverse_transform([nivel_idx])[0]
        
        return area, nivel

class AIService:
    @staticmethod
    def analisar_curriculo_ia(texto):
        """Usa um serviço de IA para análise mais aprofundada do currículo"""
        try:
            # Análise baseada em palavras-chave (substituir por IA real se necessário)
            areas_chave = ["Desenvolvimento", "Data Science", "UX/UI", "DevOps", "Gestão", "Marketing", "Financeiro"]
            niveis = ["Júnior", "Pleno", "Sênior", "Especialista"]
            
            # Determinar área com base em palavras-chave
            area_scores = {
                "Desenvolvimento": len(re.findall(r'(python|java|javascript|programação|desenvolvimento|software)', texto.lower())),
                "Data Science": len(re.findall(r'(dados|data|machine learning|estatística|análise|python|sql)', texto.lower())),
                "UX/UI": len(re.findall(r'(design|ux|ui|interface|usuario|figma|adobe)', texto.lower())),
                "DevOps": len(re.findall(r'(devops|cloud|aws|azure|docker|kubernetes|infraestrutura)', texto.lower())),
                "Gestão": len(re.findall(r'(gestão|gerência|projeto|coordenação|liderança|equipe)', texto.lower())),
                "Marketing": len(re.findall(r'(marketing|digital|mídia|redes sociais|seo|branding)', texto.lower())),
                "Financeiro": len(re.findall(r'(financeiro|contabilidade|orçamento|investimento|análise financeira)', texto.lower()))
            }
            
            area = max(area_scores.items(), key=lambda x: x[1])[0]
            
            # Determinar nível de experiência
            exp_anos = 0
            anos_match = re.search(r'(\d+)\s*anos?', texto.lower())
            if anos_match:
                exp_anos = int(anos_match.group(1))
            
            if exp_anos < 2:
                nivel = "Júnior"
            elif exp_anos < 5:
                nivel = "Pleno"
            elif exp_anos < 8:
                nivel = "Sênior"
            else:
                nivel = "Especialista"
            
            # Pontos fortes e fracos
            pontos_fortes = []
            if "python" in texto.lower():
                pontos_fortes.append("Experiência em Python")
            if any(word in texto.lower() for word in ["gestão", "liderança", "coordenação"]):
                pontos_fortes.append("Habilidades de liderança")
            if "projeto" in texto.lower():
                pontos_fortes.append("Experiência em gestão de projetos")
            if any(word in texto.lower() for word in ["aws", "azure", "google cloud"]):
                pontos_fortes.append("Conhecimento em cloud computing")
                
            pontos_fracos = []
            if len(texto) < 1000:  # Currículo muito curto
                pontos_fracos.append("Currículo pode estar incompleto")
            if not re.search(r'[\w\.-]+@[\w\.-]+\.\w+', texto):  # Sem email
                pontos_fracos.append("Informações de contato incompletas")
            
            return {
                "area_recomendada": area,
                "nivel_experiencia": nivel,
                "pontos_fortes": pontos_fortes,
                "pontos_fracos": pontos_fracos,
                "compatibilidade_geral": f"{(min(exp_anos, 10) / 10 * 100):.1f}%"
            }
        except Exception as e:
            print(f"Erro na análise de IA: {e}")
            return {
                "area_recomendada": "Desconhecida",
                "nivel_experiencia": "Desconhecido",
                "pontos_fortes": [],
                "pontos_fracos": ["Erro na análise"],
                "compatibilidade_geral": "0%"
            }

class SistemaRecrutamento:
    def __init__(self):
        self.db = GerenciadorBancoDados()
        self.classificador = ClassificadorML()
        self.requisitos = []
        self.palavras_chave_requisitos = []
    
    def adicionar_curriculo(self, caminho):
        texto = ProcessadorCurriculos.ler_arquivo(caminho)
        if not texto:
            print(f"Falha ao ler arquivo: {caminho}")
            return False
        
        nome = ProcessadorCurriculos.extrair_nome(texto)
        resumo = ProcessadorCurriculos.resumir_texto(texto)
        palavras_chave = ProcessadorCurriculos.processar_texto(texto)
        
        # Usa IA para classificar área e nível
        analise_ia = AIService.analisar_curriculo_ia(texto)
        area_interesse = analise_ia["area_recomendada"]
        nivel_experiencia = analise_ia["nivel_experiencia"]
        
        self.db.inserir_candidato(
            nome=nome,
            caminho=caminho,
            resumo=resumo,
            palavras_chave=palavras_chave,
            area_interesse=area_interesse,
            nivel_experiencia=nivel_experiencia
        )
        return True
    
    def processar_requisitos(self, texto):
        self.requisitos = [req.strip() for req in texto.split('\n') if req.strip()]
        self.palavras_chave_requisitos = ProcessadorCurriculos.processar_texto(texto)
        return self.palavras_chave_requisitos
    
    def avaliar_candidatos(self):
        if not self.palavras_chave_requisitos:
            return []
        
        candidatos = self.db.buscar_todos()
        if not candidatos:
            return []
        
        # Prepara documentos para comparação
        documentos = []
        palavras_chave_candidatos = []
        
        for c in candidatos:
            documentos.append(c[3] if c[3] else "")  # Resumos
            palavras_chave_candidatos.append(c[4].split(',') if c[4] else [])  # Palavras-chave
        
        # Adiciona os requisitos como último documento
        documentos.append('\n'.join(self.requisitos))
        
        # Calcula similaridade com TF-IDF
        try:
            vectorizer = TfidfVectorizer()
            matriz = vectorizer.fit_transform(documentos)
            similaridades = cosine_similarity(matriz[-1], matriz[:-1])[0]
        except Exception as e:
            print(f"Erro no cálculo de similaridade: {e}")
            similaridades = [0.5] * len(candidatos)  # Valor padrão em caso de erro
        
        # Calcula pontuação por palavras-chave
        palavras_chave_set = set(self.palavras_chave_requisitos)
        resultados = []
        
        for i, (id, nome, caminho, resumo, palavras_chave, area, nivel) in enumerate(candidatos):
            # Similaridade de texto
            score_texto = similaridades[i] if i < len(similaridades) else 0.5
            
            # Similaridade de palavras-chave
            palavras_candidato = palavras_chave.split(',') if palavras_chave else []
            palavras_comuns = len(palavras_chave_set.intersection(palavras_candidato))
            score_palavras = palavras_comuns / len(palavras_chave_set) if palavras_chave_set else 0
            
            # Pontuação combinada (60% texto, 40% palavras-chave)
            score_final = 0.6 * score_texto + 0.4 * score_palavras
            
            # Adiciona bônus por área e nível correspondentes
            bonus = 0
            if area and any(a.lower() in (resumo.lower() if resumo else "") for a in area.split()):
                bonus += 0.1
            if nivel and any(n.lower() in (resumo.lower() if resumo else "") for n in nivel.split()):
                bonus += 0.1
            
            score_final = min(score_final + bonus, 1.0)  # Limita a 1.0
            
            resultados.append({
                'id': id,
                'nome': nome,
                'score': score_final,
                'score_texto': score_texto,
                'score_palavras': score_palavras,
                'resumo': resumo if resumo else "Resumo não disponível",
                'palavras_chave': palavras_candidato,
                'caminho': caminho,
                'area': area if area else "Não especificada",
                'nivel': nivel if nivel else "Não especificado"
            })
        
        return sorted(resultados, key=lambda x: x['score'], reverse=True)
    
    def treinar_classificador(self):
        # Coleta dados de treinamento do banco de dados
        historico = self.db.buscar_historico_classificacao()
        
        if len(historico) < 10:  # Mínimo de exemplos para treinamento
            return False
        
        textos = []
        areas = []
        niveis = []
        
        for nome, area, nivel, data in historico:
            # Busca o texto do currículo
            candidatos = self.db.buscar_todos()
            for cand in candidatos:
                if cand[1] == nome:  # Compara nomes
                    texto_curriculo = ProcessadorCurriculos.ler_arquivo(cand[2])
                    if texto_curriculo:
                        textos.append(texto_curriculo)
                        areas.append(area)
                        niveis.append(nivel)
                    break
        
        return self.classificador.treinar_modelos(textos, areas, niveis)

class InterfaceGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Currículos - Versão IA")
        self.root.geometry("1200x800")
        
        self.sistema = SistemaRecrutamento()
        self.criar_interface()
    
    def criar_interface(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Aba principal
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Análise de Currículos")
        
        # Painel de entrada
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Requisitos da Vaga (digite cada requisito em uma linha):").pack(anchor=tk.W)
        self.txt_requisitos = tk.Text(input_frame, height=8)
        self.txt_requisitos.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Adicionar Currículo", command=self.adicionar_curriculo).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Analisar Candidatos", command=self.analisar).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Treinar Modelo IA", command=self.treinar_modelo).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Limpar Tudo", command=self.limpar).pack(side=tk.LEFT, padx=5)
        
        # Resultados
        resultados_frame = ttk.LabelFrame(main_frame, text="Resultados")
        resultados_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Treeview para resultados
        self.tree = ttk.Treeview(resultados_frame, columns=('Nome', 'Pontuação', 'Área', 'Nível', 'Texto', 'Palavras-chave'), show='headings')
        self.tree.heading('Nome', text='Nome')
        self.tree.heading('Pontuação', text='Pontuação Total')
        self.tree.heading('Área', text='Área')
        self.tree.heading('Nível', text='Nível')
        self.tree.heading('Texto', text='Similaridade Texto')
        self.tree.heading('Palavras-chave', text='Palavras-chave')
        
        self.tree.column('Nome', width=150)
        self.tree.column('Pontuação', width=80, anchor='center')
        self.tree.column('Área', width=100, anchor='center')
        self.tree.column('Nível', width=80, anchor='center')
        self.tree.column('Texto', width=80, anchor='center')
        self.tree.column('Palavras-chave', width=250)
        
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(resultados_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        
        # Detalhes
        detalhes_frame = ttk.LabelFrame(main_frame, text="Detalhes do Candidato")
        detalhes_frame.pack(fill=tk.BOTH, pady=5)
        
        self.txt_detalhes = tk.Text(detalhes_frame, height=12, wrap=tk.WORD)
        self.txt_detalhes.pack(fill=tk.BOTH, expand=True)
        
        # Configurar evento de seleção na treeview
        self.tree.bind('<<TreeviewSelect>>', self.mostrar_detalhes)
        
        # Aba de relatórios
        relatorios_frame = ttk.Frame(notebook)
        notebook.add(relatorios_frame, text="Relatórios IA")
        
        ttk.Label(relatorios_frame, text="Relatórios de Análise de IA", font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.txt_relatorios = tk.Text(relatorios_frame, height=20, wrap=tk.WORD)
        self.txt_relatorios.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(relatorios_frame, text="Gerar Relatório Completo", command=self.gerar_relatorio).pack(pady=5)
    
    def adicionar_curriculo(self):
        caminhos = filedialog.askopenfilenames(
            title="Selecione os currículos",
            filetypes=[("Documentos", "*.pdf *.docx *.txt"), ("Todos", "*.*")]
        )
        
        if caminhos:
            sucessos = 0
            for caminho in caminhos:
                if self.sistema.adicionar_curriculo(caminho):
                    sucessos += 1
            
            messagebox.showinfo("Sucesso", f"{sucessos} currículo(s) adicionado(s) com sucesso!")
    
    def analisar(self):
        requisitos = self.txt_requisitos.get("1.0", tk.END).strip()
        if not requisitos:
            messagebox.showwarning("Aviso", "Digite os requisitos da vaga")
            return
        
        self.sistema.processar_requisitos(requisitos)
        resultados = self.sistema.avaliar_candidatos()
        
        self.tree.delete(*self.tree.get_children())
        if not resultados:
            messagebox.showinfo("Info", "Nenhum currículo cadastrado")
            return
        
        for candidato in resultados:
            self.tree.insert('', tk.END, values=(
                candidato['nome'],
                f"{candidato['score']:.2f}",
                candidato['area'],
                candidato['nivel'],
                f"{candidato['score_texto']:.2f}",
                ', '.join(candidato['palavras_chave'][:5]) + ('...' if len(candidato['palavras_chave']) > 5 else '')
            ), iid=candidato['id'])
        
        # Mostra o melhor candidato
        if resultados:
            melhor = resultados[0]
            self.mostrar_detalhes_candidato(melhor)
    
    def mostrar_detalhes(self, event):
        item_selecionado = self.tree.selection()
        if not item_selecionado:
            return
        
        id_candidato = int(item_selecionado[0])
        candidatos = self.sistema.avaliar_candidatos()  # Recalcula para obter dados atualizados
        
        for candidato in candidatos:
            if candidato['id'] == id_candidato:
                self.mostrar_detalhes_candidato(candidato)
                break
    
    def mostrar_detalhes_candidato(self, candidato):
        self.txt_detalhes.config(state=tk.NORMAL)
        self.txt_detalhes.delete(1.0, tk.END)
        
        texto = f"=== {candidato['nome']} ===\n"
        texto += f"Pontuação Total: {candidato['score']:.2f}\n"
        texto += f"Área: {candidato['area']}\n"
        texto += f"Nível: {candidato['nivel']}\n"
        texto += f"Similaridade de Texto: {candidato['score_texto']:.2f}\n"
        texto += f"Match de Palavras-chave: {candidato['score_palavras']:.2f}\n\n"
        texto += f"Arquivo: {candidato['caminho']}\n\n"
        texto += "Palavras-chave encontradas:\n"
        texto += ', '.join(candidato['palavras_chave']) + "\n\n"
        texto += "Resumo:\n"
        texto += candidato['resumo']
        
        self.txt_detalhes.insert(tk.END, texto)
        self.txt_detalhes.config(state=tk.DISABLED)
    
    def treinar_modelo(self):
        if self.sistema.treinar_classificador():
            messagebox.showinfo("Sucesso", "Modelo de IA treinado com sucesso!")
        else:
            messagebox.showwarning("Aviso", "Dados insuficientes para treinar o modelo. Adicione mais currículos com classificações.")
    
    def gerar_relatorio(self):
        candidatos = self.sistema.db.buscar_todos()
        if not candidatos:
            messagebox.showinfo("Info", "Nenhum currículo cadastrado")
            return
        
        relatorio = "=== RELATÓRIO DE ANÁLISE DE CURRÍCULOS ===\n\n"
        relatorio += f"Total de currículos: {len(candidatos)}\n\n"
        
        # Estatísticas por área
        areas = defaultdict(int)
        niveis = defaultdict(int)
        
        for cand in candidatos:
            if cand[5]:  # área
                areas[cand[5]] += 1
            if cand[6]:  # nível
                niveis[cand[6]] += 1
        
        relatorio += "Distribuição por Área:\n"
        for area, count in areas.items():
            relatorio += f"  {area}: {count} candidatos\n"
        
        relatorio += "\nDistribuição por Nível:\n"
        for nivel, count in niveis.items():
            relatorio += f"  {nivel}: {count} candidatos\n"
        
        # Top habilidades
        todas_palavras_chave = []
        for cand in candidatos:
            if cand[4]:  # palavras-chave
                todas_palavras_chave.extend(cand[4].split(','))
        
        from collections import Counter
        contador = Counter(todas_palavras_chave)
        top_habilidades = contador.most_common(10)
        
        relatorio += "\nTop 10 Habilidades:\n"
        for habilidade, count in top_habilidades:
            relatorio += f"  {habilidade}: {count} ocorrências\n"
        
        self.txt_relatorios.delete(1.0, tk.END)
        self.txt_relatorios.insert(tk.END, relatorio)
    
    def limpar(self):
        self.txt_requisitos.delete("1.0", tk.END)
        self.tree.delete(*self.tree.get_children())
        self.txt_detalhes.config(state=tk.NORMAL)
        self.txt_detalhes.delete("1.0", tk.END)
        self.txt_detalhes.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceGrafica(root)
    root.mainloop()
