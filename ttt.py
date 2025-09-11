import re
import sqlite3
from pathlib import Path
import pdfplumber
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import Counter, defaultdict
import logging
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("recrutamento.log"),
        logging.StreamHandler()
    ]
)

# Carregamento do modelo Spacy
try:
    nlp = spacy.load("pt_core_news_sm")
except:
    logging.error("Instale o modelo de linguagem: python -m spacy download pt_core_news_sm")
    exit()

class GerenciadorBancoDados:
    def __init__(self, db_name='recrutamento.db'):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.criar_tabelas()
    
    def criar_tabelas(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidatos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            caminho_curriculo TEXT,
            resumo TEXT,
            data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.commit()

        # Verificação e adição de colunas ausentes
        cursor.execute("PRAGMA table_info(candidatos)")
        colunas = [info[1] for info in cursor.fetchall()]

        if 'palavras_chave' not in colunas:
            cursor.execute("ALTER TABLE candidatos ADD COLUMN palavras_chave TEXT")
        if 'area_interesse' not in colunas:
            cursor.execute("ALTER TABLE candidatos ADD COLUMN area_interesse TEXT")
        if 'nivel_experiencia' not in colunas:
            cursor.execute("ALTER TABLE candidatos ADD COLUMN nivel_experiencia TEXT")
        
        self.conn.commit()
    
    def inserir_candidato(self, nome, caminho, resumo, palavras_chave):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO candidatos (nome, caminho_curriculo, resumo, palavras_chave)
            VALUES (?, ?, ?, ?)''',
            (nome.strip(), caminho, resumo, ','.join(palavras_chave)))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logging.error(f"Erro ao inserir candidato: {e}")
            return None
    
    def buscar_todos(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, nome, caminho_curriculo, resumo, palavras_chave, area_interesse, nivel_experiencia FROM candidatos')
        return cursor.fetchall()
        
    def buscar_candidato_por_nome(self, nome):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM candidatos WHERE nome = ?', (nome,))
        return cursor.fetchone()
        
    def deletar_candidato(self, id):
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM candidatos WHERE id = ?', (id,))
            self.conn.commit()
            logging.info(f"Candidato com ID {id} deletado com sucesso.")
            return True
        except Exception as e:
            logging.error(f"Erro ao deletar candidato: {e}")
            return False


class ProcessadorCurriculos:
    @staticmethod
    def ler_arquivo(caminho):
        try:
            extensao = Path(caminho).suffix.lower()
            if extensao == '.txt':
                with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif extensao == '.pdf':
                texto = ''
                with pdfplumber.open(caminho) as pdf:
                    for pagina in pdf.pages:
                        page_text = pagina.extract_text()
                        if page_text:
                            texto += page_text + '\n'
                return texto
            elif extensao == '.docx':
                doc = docx.Document(caminho)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logging.error(f"Erro ao ler arquivo {caminho}: {e}")
            return None

    @staticmethod
    def extrair_palavras_chave(texto, top_k=15):
        if not texto:
            return []
        doc = nlp(texto)
        counts = Counter()
        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.lemma_) > 2:
                counts[token.lemma_.lower()] += 1
        return [w for w, _ in counts.most_common(top_k)]

class SistemaRecrutamento:
    def __init__(self):
        self.db = GerenciadorBancoDados()
        self.requisitos = None
        self.modelo = None
        self.label_encoder = LabelEncoder()

    def adicionar_curriculo(self, caminho):
        texto = ProcessadorCurriculos.ler_arquivo(caminho)
        if not texto:
            return False
        nome = Path(caminho).stem
        resumo = texto
        palavras_chave = ProcessadorCurriculos.extrair_palavras_chave(texto)
        self.db.inserir_candidato(nome, caminho, resumo, palavras_chave)
        return True

    def processar_requisitos(self, texto):
        self.requisitos = texto.strip()
        return True

    def avaliar_candidatos(self):
        candidatos = self.db.buscar_todos()
        if not candidatos or not self.requisitos:
            return []
        docs = [resumo for _, _, _, resumo, _, _, _ in candidatos]
        try:
            vectorizer = TfidfVectorizer(max_features=2000)
            matriz = vectorizer.fit_transform(docs + [self.requisitos])
            req_vec = matriz[-1]
            docs_vecs = matriz[:-1]
            scores = cosine_similarity(req_vec, docs_vecs)[0]
        except Exception as e:
            logging.error(f"Erro na análise TF-IDF: {e}")
            scores = [0.0] * len(docs)
        resultados = []
        for i, (id, nome, caminho, resumo, palavras, area, nivel) in enumerate(candidatos):
            resultados.append({
                'id': id,
                'nome': nome,
                'score': float(scores[i]),
                'resumo': resumo[:300],
                'palavras_chave': palavras.split(',') if palavras else [],
                'area': area or 'Não definida',
                'nivel': nivel or 'Não definido'
            })
        resultados.sort(key=lambda x: x['score'], reverse=True)
        return resultados

    def treinar_ia(self):
        candidatos = self.db.buscar_todos()
        if not candidatos:
            return False
        X = [resumo for _, _, _, resumo, _, _, _ in candidatos]
        y = [area if area else "Outros" for _, _, _, _, _, area, _ in candidatos]
        if len(set(y)) < 2:
            return False
        vectorizer = TfidfVectorizer(max_features=2000)
        X_vec = vectorizer.fit_transform(X)
        y_enc = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)
        self.modelo = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)
        self.modelo.fit(X_train, y_train)
        score = self.modelo.score(X_test, y_test)
        logging.info(f"Acurácia do modelo: {score:.2f}")
        return True

class InterfaceGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Currículos - IA Avançada")
        self.root.geometry("1200x800")
        self.sistema = SistemaRecrutamento()
        self.criar_interface()

    def criar_interface(self):
        style = ttk.Style()
        style.configure("Treeview", font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Análise de Currículos")

        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Relatórios")

        frame1 = ttk.Frame(tab1)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Label(frame1, text="Requisitos da Vaga:", font=("Segoe UI", 11)).pack(anchor=tk.W)
        self.txt_requisitos = tk.Text(frame1, height=5, font=("Segoe UI", 10))
        self.txt_requisitos.pack(fill=tk.X, pady=5)

        frame2 = ttk.Frame(tab1)
        frame2.pack(fill=tk.X)
        ttk.Button(frame2, text="Adicionar Currículo", command=self.adicionar_curriculo).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame2, text="Analisar", command=self.analisar).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame2, text="Deletar Candidato", command=self.deletar_candidato).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame2, text="Treinar IA", command=self.treinar_ia).pack(side=tk.LEFT, padx=5)


        self.tree = ttk.Treeview(tab1, columns=("Nome", "Pontuação", "Área", "Nível"), show="headings")
        self.tree.heading("Nome", text="Nome")
        self.tree.heading("Pontuação", text="Pontuação")
        self.tree.heading("Área", text="Área")
        self.tree.heading("Nível", text="Nível")
        self.tree.column("Nome", width=200)
        self.tree.column("Pontuação", width=100, anchor='center')
        self.tree.column("Área", width=150, anchor='center')
        self.tree.column("Nível", width=150, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Adiciona o evento de clique na Treeview
        self.tree.bind("<ButtonRelease-1>", self.exibir_detalhes_candidato)

        self.txt_relatorios = tk.Text(tab2, height=20, font=("Segoe UI", 10))
        self.txt_relatorios.pack(fill=tk.BOTH, expand=True)

        ttk.Button(tab2, text="Gerar Relatório", command=self.gerar_relatorio).pack(pady=5)
    
    def deletar_candidato(self):
        item_id = self.tree.focus()
        if not item_id:
            messagebox.showwarning("Aviso", "Por favor, selecione um candidato para deletar.")
            return

        valores = self.tree.item(item_id, 'values')
        nome_candidato = valores[0]
        
        # Confirmação de exclusão
        confirmacao = messagebox.askyesno(
            "Confirmação de Exclusão",
            f"Você tem certeza que deseja deletar o candidato '{nome_candidato}'?"
        )
        if not confirmacao:
            return

        candidato = self.sistema.db.buscar_candidato_por_nome(nome_candidato)
        if not candidato:
            messagebox.showwarning("Erro", "Candidato não encontrado no banco de dados.")
            return
        
        if self.sistema.db.deletar_candidato(candidato[0]):
            self.tree.delete(item_id)
            messagebox.showinfo("Sucesso", "Candidato deletado com sucesso.")
        else:
            messagebox.showerror("Erro", "Ocorreu um erro ao tentar deletar o candidato.")


    def adicionar_curriculo(self):
        caminhos = filedialog.askopenfilenames(
            title="Selecione os currículos",
            filetypes=[("Documentos", "*.pdf *.docx *.txt"), ("Todos", "*.*")]
        )
        if caminhos:
            for caminho in caminhos:
                self.sistema.adicionar_curriculo(caminho)
            messagebox.showinfo("Sucesso", "Currículos adicionados!")

    def analisar(self):
        requisitos = self.txt_requisitos.get("1.0", tk.END).strip()
        if not requisitos:
            messagebox.showwarning("Aviso", "Por favor, insira os requisitos da vaga para análise.")
            return

        self.sistema.processar_requisitos(requisitos)
        resultados = self.sistema.avaliar_candidatos()
        
        self.tree.delete(*self.tree.get_children())
        if not resultados:
            messagebox.showinfo("Aviso", "Nenhum candidato encontrado ou não há requisitos definidos.")
            return

        for c in resultados:
            self.tree.insert('', tk.END, values=(c['nome'], f"{c['score']:.2f}", c['area'], c['nivel']))
            
    def exibir_detalhes_candidato(self, event):
        item_id = self.tree.focus() # Obtém o ID do item focado
        if not item_id:
            return
        
        # Obtém os valores da linha selecionada
        valores = self.tree.item(item_id, 'values')
        nome_candidato = valores[0]
        
        # Busca o candidato no banco de dados pelo nome
        candidato = self.sistema.db.buscar_candidato_por_nome(nome_candidato)
        if not candidato:
            messagebox.showwarning("Erro", "Detalhes do candidato não encontrados.")
            return
        
        # Cria uma nova janela para exibir os detalhes
        detalhes_janela = tk.Toplevel(self.root)
        detalhes_janela.title(f"Detalhes de {candidato[1]}")
        detalhes_janela.geometry("600x400")
        
        frame_detalhes = ttk.Frame(detalhes_janela, padding="10")
        frame_detalhes.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame_detalhes, text=f"Nome: {candidato[1]}", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Adiciona um campo de texto para o resumo do currículo
        ttk.Label(frame_detalhes, text="Resumo do Currículo:", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(5, 0))
        txt_resumo = tk.Text(frame_detalhes, wrap=tk.WORD, font=("Segoe UI", 10))
        txt_resumo.insert(tk.END, candidato[3])
        txt_resumo.pack(fill=tk.BOTH, expand=True)
        txt_resumo.config(state=tk.DISABLED) # Torna o texto somente leitura
        
        # Adiciona as palavras-chave, área e nível
        ttk.Label(frame_detalhes, text=f"Palavras-chave: {candidato[4]}", font=("Segoe UI", 11)).pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(frame_detalhes, text=f"Área de Interesse: {candidato[5] or 'Não definida'}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(frame_detalhes, text=f"Nível de Experiência: {candidato[6] or 'Não definido'}", font=("Segoe UI", 11)).pack(anchor=tk.W)


    def treinar_ia(self):
        if self.sistema.treinar_ia():
            messagebox.showinfo("Sucesso", "IA treinada com sucesso!")
        else:
            messagebox.showwarning("Aviso", "Não há dados suficientes para treinar a IA.")

    def gerar_relatorio(self):
        candidatos = self.sistema.db.buscar_todos()
        if not candidatos:
            messagebox.showinfo("Aviso", "Nenhum candidato cadastrado para gerar relatório.")
            return

        relatorio = f"Total de currículos: {len(candidatos)}\n\n"
        areas = defaultdict(int)
        niveis = defaultdict(int)
        todas_palavras_chave = []
        for cand in candidatos:
            if cand[5]:
                areas[cand[5]] += 1
            if cand[6]:
                niveis[cand[6]] += 1
            if cand[4]:
                todas_palavras_chave.extend(cand[4].split(','))
        
        relatorio += "Distribuição por Área:\n"
        for area, count in areas.items():
            relatorio += f"  {area}: {count} candidatos\n"
        
        relatorio += "\nDistribuição por Nível:\n"
        for nivel, count in niveis.items():
            relatorio += f"  {nivel}: {count} candidatos\n"
        
        contador = Counter(todas_palavras_chave)
        top_habilidades = contador.most_common(10)
        
        relatorio += "\nTop 10 Habilidades:\n"
        for habilidade, count in top_habilidades:
            relatorio += f"  {habilidade}: {count} ocorrências\n"
        
        self.txt_relatorios.delete(1.0, tk.END)
        self.txt_relatorios.insert(tk.END, relatorio)

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceGrafica(root)
    root.mainloop()
