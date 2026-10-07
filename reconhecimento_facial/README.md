# Reconhecimento facial com OpenCV e DeepFace

Solucao em **script Python**, sem Jupyter Notebook. O OpenCV le e salva as imagens,
desenha os bounding boxes e escreve os textos. O DeepFace detecta/alinha os rostos,
extrai suas representacoes com Facenet512 e compara cada rosto com os cadastros.
Tambem e possivel usar a webcam do notebook para verificar os rostos continuamente.

## Estrutura

```text
reconhecimento_facial/
├── main.py
├── requirements.txt
├── README.md
├── tests/
│   ├── test_camera.py
│   └── test_reconhecimento.py
├── fotos/
│   ├── autorizados/
│   │   ├── Ana/             # Crie uma pasta por pessoa (recomendado).
│   │   │   ├── foto1.jpg
│   │   │   └── foto2.jpg
│   │   └── Bruno/
│   │       └── foto1.jpg
│   └── testes/
│       ├── pessoa_autorizada.jpg
│       ├── pessoa_desconhecida.jpg
│       └── grupo.jpg
└── resultados/
    ├── pessoa_autorizada.jpg.png
    ├── pessoa_desconhecida.jpg.png
    ├── grupo.jpg.png
    └── relatorio.json
```

As pastas de entrada e de resultados foram criadas vazias. Os nomes de pessoas e
fotos acima sao exemplos: adicione suas fotos para executar o reconhecimento.
Use arquivos JPG, JPEG, PNG ou BMP. Subpastas sao processadas recursivamente.

## Usar o ambiente ja preparado

Se a pasta `.venv` deste projeto ja existe e as dependencias estao instaladas,
nao execute novamente `py -3.12 -m venv .venv`. A partir da raiz `MLsample`, use:

```powershell
.\reconhecimento_facial\.venv\Scripts\python.exe .\reconhecimento_facial\main.py
```

Esse comando usa diretamente o ambiente do reconhecimento facial, mesmo que
outro ambiente, como `(module-olist)`, esteja ativo no terminal. As fotos devem
estar nas pastas de cadastro e testes descritas abaixo.

No **Git Bash**, estando dentro de `reconhecimento_facial/`, use:

```bash
./.venv/Scripts/python.exe main.py
```

## Criar um novo ambiente no Windows (PowerShell)

Use Python 3.11 ou 3.12 de 64 bits para o conjunto de dependencias especificado.
Este procedimento so e necessario se o ambiente do projeto ainda nao existe.
O comando `py -0p` lista as versoes reconhecidas pelo launcher do Windows. Se ele
nao reconhecer Python 3.12, use o caminho completo de um Python 3.11/3.12 disponivel
para criar o ambiente, ou instale uma dessas versoes antes de continuar.
Na raiz do projeto, execute:

```powershell
cd reconhecimento_facial
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se usar Python 3.11, troque `-3.12` por `-3.11`. Se o ambiente `.venv` ja estiver
criado, execute diretamente o comando de instalacao das dependencias.
Nao e necessario ativar o ambiente virtual. Na primeira execucao, o DeepFace baixa
os pesos do modelo; essa etapa precisa de internet e pode levar alguns minutos.
O tf-keras esta incluido para compatibilidade com o TensorFlow 2.16. Por padrao,
os pesos sao guardados em `.deepface/` dentro desta pasta do projeto.

## Cadastro e execucao

1. Coloque fotos de pessoas autorizadas em `fotos/autorizados/NomeDaPessoa/`.
   Cada foto deve conter **exatamente um rosto**, visivel e bem iluminado.
   Fotos diretamente em `autorizados/` tambem funcionam; o nome do arquivo vira
   o identificador da pessoa. Para varias fotos da mesma pessoa, use uma subpasta.
2. Coloque fotos independentes de teste em `fotos/testes/`. Inclua uma pessoa
   cadastrada, uma pessoa nao cadastrada e uma foto com varios rostos.
3. Execute:

```powershell
.\.venv\Scripts\python.exe main.py
```

4. Abra as imagens geradas em `resultados/`:
   - Rosto reconhecido: bounding box **verde** com **ACESSO LIBERADO**.
   - Rosto desconhecido: bounding box **vermelho** com **ACESSO NEGADO**.
   - Cada rosto de uma foto com varias pessoas recebe uma decisao individual.
   - Sem rosto detectado, a imagem recebe uma borda vermelha com **ACESSO NEGADO**.
5. Consulte `resultados/relatorio.json` para coordenadas, identidade reconhecida,
   distancia, limiar e erros de processamento.

As imagens originais sao preservadas. Executar novamente atualiza as imagens de
saida de mesmo nome e o relatorio. Saidas antigas de testes removidos permanecem
na pasta; o relatorio lista somente a execucao atual. Cadastros invalidos interrompem
a inicializacao. Erros de comparacao negam acesso. Arquivos ilegíveis sao registrados
como erro e o processamento continua nos demais. O codigo de saida e 1 quando ha
falha de processamento (inclusive ausencia de rosto), e 0 quando termina sem falhas;
uma pessoa desconhecida detectada e um resultado normal, nao um erro.

## Camera em tempo real

Cadastre primeiro as fotos em `fotos/autorizados/NomeDaPessoa/`, com exatamente
um rosto por foto. No modo de camera, nao e necessario ter fotos em `fotos/testes/`.

No **Git Bash**, dentro da pasta `reconhecimento_facial/`:

```bash
./.venv/Scripts/python.exe main.py --camera
```

No **PowerShell**, dentro da mesma pasta:

```powershell
.\.venv\Scripts\python.exe main.py --camera
```

O programa carrega os cadastros e abre uma janela com dois paineis: **AO VIVO**,
com o video continuo e o FPS medido, e **ULTIMA ANALISE**, com o quadro que foi
reconhecido e sua idade em segundos. Cada rosto do quadro analisado recebe uma
caixa verde com **ACESSO LIBERADO** ou vermelha com **ACESSO NEGADO**. Quando
nenhum rosto e detectado ou a analise falha, o quadro analisado recebe acesso negado.
Pressione **Q** ou **Esc**, com a janela selecionada, para sair. Tambem e possivel
fechar a janela pelo X ou interromper com Ctrl+C no terminal; a camera e liberada.

Se a camera do notebook nao abrir, feche outros aplicativos que usam a webcam,
confira as permissoes de camera para aplicativos de desktop nas configuracoes do
Windows ou tente outro indice:

```bash
./.venv/Scripts/python.exe main.py --camera --indice-camera 1
```

A captura e a exibicao continuam enquanto o DeepFace trabalha em uma thread
separada. O programa solicita 640x480 a 30 FPS e analisa imagens de ate 480 pixels
de largura, com intervalo minimo de 0.25 segundo entre envios. Existe apenas uma
analise em andamento: quando ela termina, o proximo envio utiliza o quadro atual,
sem acumular uma fila de imagens antigas. As caixas ficam no quadro analisado,
para que uma autorizacao anterior nao seja aplicada a outro rosto no video ao vivo.

Para reduzir mais o custo da analise, experimente no Git Bash:

```bash
./.venv/Scripts/python.exe main.py --camera --largura-analise 320 --intervalo-analise 0.5
```

Imagens menores podem dificultar a deteccao de rostos distantes. Aumentar o intervalo
reduz a frequencia das verificacoes. `--fps-camera 30` configura o FPS solicitado;
o FPS real depende da webcam, iluminacao e computador, e aparece no painel ao vivo.
O FPS do video e a frequencia de reconhecimento sao diferentes: as verificacoes
tambem dependem do tempo que o DeepFace leva para analisar cada imagem.
O modo de camera exibe o resultado na janela, sem gravar video, fotos ou relatorio.
Os cadastros sao recarregados ao reiniciar o programa.

## Opcoes

Os caminhos padrao sao relativos a `main.py`. Caminhos passados por argumento sao
relativos ao diretorio de onde o comando foi executado.

```powershell
.\.venv\Scripts\python.exe main.py --autorizados "C:\fotos\autorizados" --testes "C:\fotos\testes" --saida "C:\fotos\resultados"
.\.venv\Scripts\python.exe main.py --modelo Facenet512 --limiar 0.25
```

Sem `--limiar`, o programa utiliza o limiar do DeepFace para o modelo e a distancia
cosseno. Uma distancia menor representa maior semelhanca. O valor 0.25 acima e
apenas um exemplo de configuracao; avalie o limiar com suas fotos de teste.
As referencias sao calculadas uma vez por execucao e ficam em memoria.

## Verificacao

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes verificam as decisoes de acesso, falhas, cores dos bounding boxes e
encerramento/liberacao da camera usando
respostas simuladas do DeepFace, sem baixar modelos. Eles exigem as dependencias
instaladas e nao substituem uma execucao com fotos reais. Para verificar o modelo,
execute os tres cenarios de fotos descritos acima e confira os resultados visuais.

Esta implementacao avalia correspondencia facial em fotografias; o texto de acesso
e o resultado dessa comparacao. Ela nao aciona fechaduras nem verifica presenca viva.

Referencias: [DeepFace 0.0.93](https://github.com/serengil/deepface/tree/v0.0.93)
e [OpenCV](https://docs.opencv.org/4.x/).
Captura da webcam: [tutorial oficial de video do OpenCV](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html).
