# BoneCheck

BoneCheck é um projeto de inteligência artificial voltado para a detecção precoce de osteoporose por meio da análise automática de radiografias panorâmicas utilizando redes neurais artificiais. O sistema foi desenvolvido para auxiliar profissionais da saúde na triagem e identificação de pacientes com risco elevado de osteoporose, oferecendo uma alternativa acessível, rápida e não invasiva ao exame tradicional de densitometria óssea.

## Qual problema ele resolve?
A osteoporose é uma doença silenciosa e progressiva, caracterizada pela perda de massa óssea e aumento do risco de fraturas. Frequentemente, ela só é diagnosticada após a ocorrência de uma fratura grave, o que já representa um estágio avançado da doença. O problema central é a falta de diagnóstico precoce, especialmente em regiões com acesso limitado a exames específicos como a densitometria óssea (DEXA).

BoneCheck propõe resolver essa lacuna ao utilizar radiografias panorâmicas, exames já comuns na odontologia, para detectar sinais indicativos da doença com o auxílio de inteligência artificial. Isso permite o aproveitamento de exames já realizados, sem a necessidade de exames adicionais, ampliando a capacidade de detecção precoce.

Feito em parceria com a FORB - USP Ribeirão Preto, com os Prof. Dr. Plauto Watanabe e Prof. Dra. Luciana Munhoz.


## ⚙️ Funcionamento
BoneCheck utiliza um pipeline de classificação de imagens médicas com modelos de redes neurais convolucionais (CNNs) e uma etapa final de fusão de predições com XGBoost, um algoritmo de aprendizado de máquina baseado em árvores de decisão. O sistema foi projetado para analisar radiografias panorâmicas e classificar entre Saudável, Osteopenia e Osteoporose.

## 🔄 Pipeline de Funcionamento do BoneCheck

![bone_check](https://github.com/user-attachments/assets/7127070a-7d18-4110-9262-57fa21db6cda)


## 🛠️ Como rodar
Clone o repositório, treine as 4 redes neurais, treine o XGBoost, utilize a aplicação final!

### Clonando repositório
```bash
git clone https://github.com/gruporaia/BoneCheck.git
cd BoneCheck
```

### Instalando dependências
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Executando o projeto
```bash
python -m cnn.train_holdout                                  # Treina o modelo
python -m cnn.tensorboard.main --logdir outputs --port 6006  # Monitore o treinamento
python -m xgb.cnn_xgb                                        # Treina o XGBoost
streamlit run app.py                                         # Abre a interface web
```

## 📊 Resultados
| Modelo         | Acurácia | Precisão | Recall | F1-Score |
|----------------|----------|----------|--------|----------|
| **ConvNeXT**   | 0.688    | 0.651    | 0.631  | 0.638    |
| **EfficientNet** | 0.558  | 0.608    | 0.605  | 0.512    |
| **DeiT**       | 0.688    | 0.648    | **0.635**  | 0.639    |
| **Swin**       | 0.632    | 0.591    | 0.622  | 0.590    |
| **Ensemble**   | **0.697**    |**0.712**    | 0.622  | **0.838**    |
| **GPT-4o**     | 0.372    | 0.555    | 0.358  | 0.261    |
| **Gemini 2.5 Pro** | 0.602 | 0.517    | 0.487  | 0.484    |


### Próximos passos 
* Testar *ensembling* mais avançadas, como stacking com validação cruzada e modelos meta-aprendizes.
* Adicionar feedback visual das áreas da imagem que influenciaram a predição (*heatmaps*, Grad-CAM).
* Desenvolver uma interface mais amigável, com foco em usabilidade para profissionais da saúde (ex.: dentistas, clínicos gerais).


## 📑 Referências
### 📝 Artigos Científicos
- Gao, W. et al. (2023). [Artificial intelligence model based on panoramic radiographs for early diagnosis of osteoporosis](https://link.springer.com/article/10.1007/s40846-023-00831-x). *Journal of Medical and Biological Engineering*.
- Lee, J. H. et al. (2022). [Deep learning based osteoporosis detection using panoramic dental radiographs](https://europepmc.org/article/med/36185321). *Scientific Reports*.
- Wang, Z. et al. (2024). [Diagnosis of osteoporosis using panoramic radiographs and AI techniques](https://journals.sagepub.com/doi/epub/10.1177/03000605241274576). *Journal of International Medical Research*.
- Liu, X. et al. (2024). [Explainable AI in dental imaging for osteoporosis risk classification](https://journals.sagepub.com/doi/epub/10.1177/03000605241244754). *Journal of International Medical Research*.
- Silva, M. et al. (2025). [Explainable deep learning for osteoporosis screening in dental X-rays](https://www.sciencedirect.com/science/article/pii/S0300571225000958?via%3Dihub). *Building and Environment*.
- Zhang, Y. et al. (2024). [Multi-task transformer for bone quality prediction](https://www.sciencedirect.com/science/article/pii/S1746809424010899). *Biomedical Signal Processing and Control*.

### 📚 Artigos Técnicos e Arquiteturais
- Vaswani, A. et al. (2017). [Attention is All You Need](https://arxiv.org/abs/1703.05175). *arXiv preprint arXiv:1703.05175*.
- Tan, M., & Le, Q. V. (2019). [EfficientNet: Rethinking model scaling for convolutional neural networks](https://arxiv.org/abs/1905.11946). *arXiv preprint arXiv:1905.11946*.
- Dosovitskiy, A. et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). *arXiv preprint arXiv:2010.11929*.
- Chen, T., & Guestrin, C. (2016). [XGBoost: A scalable tree boosting system](https://arxiv.org/abs/1603.02754). *arXiv preprint arXiv:1603.02754*.

### 🧰 Ferramentas e Tutoriais
- Google Cloud. [Vertex AI: Multimodal Image Understanding](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-un).


## 💻 Quem somos
| ![LogoRAIA](https://github.com/user-attachments/assets/ce3f8386-a900-43ff-af84-adce9c17abd2) |  Este projeto foi desenvolvido pelos membros do **RAIA (Rede de Avanço de Inteligência Artificial)**, uma iniciativa estudantil do Instituto de Ciências Matemáticas e de Computação (ICMC) da USP - São Carlos. Somos estudantes que compartilham o objetivo de criar soluções inovadoras utilizando inteligência artificial para impactar positivamente a sociedade. Para saber mais, acesse [nosso site](https://gruporaia.vercel.app/) ou [nosso Instagram](instagram.com/grupo.raia)! |
|------------------|-------------------------------------------|

### Desenvolvedores
- **Andre De Mitri** - [LinkedIn](https://www.linkedin.com/in/pedroamdelgado) | [GitHub](https://github.com/andregdmitri)
- **Ademir Guimarães** - [LinkedIn](https://www.linkedin.com/in/ademir-guimaraes) | [GitHub](https://github.com/demiguic)
- **Matheus Giraldi** - [LinkedIn](https://www.linkedin.com/in/matheus-giraldi-alvarenga-b2b856217) | [GitHub](https://github.com/matheusgiraldi)
- **Matheus Lenzi** - [LinkedIn](https://www.linkedin.com/in/matheus-lenzi-dos-santos) | [GitHub](https://github.com/Matheus-Lenzi)
- **Yasmin Oliveira** - [LinkedIn](https://www.linkedin.com/in/yasmin-victoria-oliveira) | [GitHub](https://github.com/yasminvo)
- **Gabriel Merlin** - [LinkedIn](https://www.linkedin.com/in/gabrielcmerlin) | [GitHub](https://github.com/gabrielcmerlin)

Agradecimentos especiais aos Prof Dr. Plauto Watanabe e Dra. Luciana Munhoz.
