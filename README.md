# Playsic

Pipeline de deep learning que recebe **qualquer música** e produz
automaticamente uma **pasta jogável no Clone Hero** (jogo rítmico estilo
Guitar Hero).

A entrada é um arquivo de áudio (`.mp3`, `.wav`, `.ogg`, ...) e o BPM da
música. A saída é uma pasta no formato Clone Hero — mapa de notas
(`notes.mid`), áudios separados por instrumento, capa do álbum e metadados
(`song.ini`) — pronta pra arrastar pra `<CloneHero>/Songs/` e jogar.

A geração das notas usa dois motores de inferência, escolhidos por instrumento:

- **guitar / bass (rhythm) / vocals** → **basic-pitch** (modelo pré-treinado de
  transcrição áudio→MIDI, backend ONNX). Os pitches transcritos viram frets do
  Clone Hero por **contorno melódico** (a melodia sobe → fret mais alto); vocals
  recebe os pitches reais cantados.
- **drums** → modelo **CRNN** treinado (`DrumCRNN`), já que basic-pitch não
  transcreve bateria.

O `notes.mid` final é gerado com **redução real por dificuldade**
(Easy < Medium < Hard < Expert) — cada nível inferior é um subconjunto afinado
do Expert (menos notas + acordes simplificados), não uma cópia.

---

## Pipeline principal

```
musica.mp3 + BPM
      │
      ▼
separa_audio.py  (Demucs htdemucs_6s)
      │
      ▼
drums.ogg, guitar.ogg, rhythm.ogg, vocals.ogg, song.ogg
      │
      ├── drums.ogg  ──►  CRNN (DrumCRNN) ──► aba 'drums'  ──┐
      ├── guitar.ogg ──►  basic-pitch     ──► aba 'guitar' ──┤
      ├── rhythm.ogg ──►  basic-pitch     ──► aba 'rhythm' ──┤
      └── vocals.ogg ──►  basic-pitch     ──► aba 'vocals' ──┘
                                                              │
                                                              ▼
                                                  notes.xlsx consolidado
                                                              │
                                                              ▼
                              excel_to_midi.py → notes.mid (4 dificuldades)
                                                              │
                                                              ▼
                                              song_ini.py     → song.ini
                                                              │
                                                              ▼
                                  resultados/charts/<nome>/   ← pasta jogável
                                                              │
                                                              ▼
                                          onyx_web_preview.py → index.html
                                                              │
                                                              ▼
                                  resultados/previews/<nome>/ ← preview no browser
```

Observações:

- `song.ogg` é **obrigatório** para o Clone Hero. É a mixagem dos 5 stems
  não-vocais do Demucs (guitar + bass + drums + piano + other).
- Os 4 modelos rodam em paralelo sobre os 4 `.ogg` de instrumento.
- Cada modelo produz um `.xlsx` parcial com aba `info` + aba do
  instrumento. Os 4 são consolidados num único `notes.xlsx` antes do
  `excel_to_midi.py`.
- O BPM é informado pelo usuário (entrada do `main.py` futuro). Os modelos
  dependem dele para calcular o grid de tempo.

---

## Estrutura do repositório

```
playsic/
├── README.md
├── requirements.txt
├── .gitignore
├── song_ini.py                    # áudio → song.ini
│
├── dados/                         # IMUTÁVEL — não alterar estrutura
│   ├── original/                  # arquivos Rock Band brutos
│   ├── pre_dataset/               # saída do onyx_rb_to_ch
│   ├── dataset/                   # dataset usado no treinamento
│   └── organizar_dataset.py
│
├── onyx/
│   ├── file/                      # onyx-*-linux-x64.AppImage aqui
│   ├── onyx_rb_to_ch.py
│   └── onyx_web_preview.py
│
├── processamento/
│   ├── audio/
│   │   └── separa_audio.py
│   └── midi_excel/
│       ├── midi_to_excel.py
│       └── excel_to_midi.py
│
├── treinamento/
│   ├── drum_crnn.py                       # arquitetura do modelo
│   ├── audio_features.py                  # mel-espectrograma alinhado ao grid
│   ├── notes_xlsx.py                      # leitura/escrita do xlsx resumido
│   ├── training_utils.py                  # Dataset, preprocess, augment
│   ├── modelo_gera_excel.py               # inferência: áudio + bpm → xlsx
│   ├── treinamento_modelo_drums.ipynb
│   ├── checkpoint/{drums,guitar,bass,vocals}/
│   ├── logs/{drums,guitar,bass,vocals}/
│   ├── modelos/
│   └── validação/{drums,guitar,bass,vocals}/
│
└── resultados/
    ├── charts/                    # pastas jogáveis
    └── previews/                  # previews HTML
```

---

## Instalação

**Sistema** (Ubuntu/Debian):
```bash
sudo apt install ffmpeg
```

**Python** (3.10+):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Binário externo — Onyx CLI:**
Necessário apenas para converter Rock Band → Clone Hero (preparo de dataset)
e gerar previews HTML.

1. Baixar o AppImage Linux mais recente em
   [github.com/mtolly/onyx/releases](https://github.com/mtolly/onyx/releases).
2. Salvar em `onyx/file/onyx-*-linux-x64.AppImage`.
3. Os scripts extraem automaticamente em runtime (sem precisar de FUSE).

Alternativas: definir `$ONYX_CLI` apontando para um binário Onyx já
instalado, ou ter `onyx` no `$PATH`.

---

## Como usar cada script

Todos os scripts têm **dupla interface**: função pública importável + CLI
com `argparse`.

### Preparo de dataset

```bash
# 1) Converte arquivos Rock Band (CON/LIVE/PIRS/.pkg/.rba) para Clone Hero
python onyx/onyx_rb_to_ch.py \
    --input dados/original/ \
    --output dados/pre_dataset/

# 2) Monta dados/dataset/ + dataset_metadata.xlsx
python dados/organizar_dataset.py \
    --src dados/pre_dataset/ \
    --dst dados/dataset/
```

### Treinamento

Abrir `treinamento/treinamento_modelo_drums.ipynb` no Jupyter e rodar. Ao
final, o notebook gera automaticamente o chart + preview de uma música do
val set para inspeção.

### Inferência standalone

```bash
python treinamento/modelo_gera_excel.py \
    --audio drums.ogg \
    --bpm 140 \
    --instrument drums \
    --model treinamento/checkpoint/drums/drums_crnn_best.pt \
    --meta  treinamento/checkpoint/drums/drums_crnn_meta.pt \
    --out   drums_partial.xlsx
```

### Utilitários

```bash
# Separação de stems (Demucs)
python processamento/audio/separa_audio.py --audio musica.mp3 --out stems/

# MIDI → planilha detalhada (análise)
python processamento/midi_excel/midi_to_excel.py --midi notes.mid --out notes.xlsx

# Planilha resumida → MIDI
python processamento/midi_excel/excel_to_midi.py --xlsx notes.xlsx --out notes.mid

# song.ini a partir das tags do áudio original
python song_ini.py --audio musica.mp3 --out resultados/charts/Foo/song.ini

# Web preview (uma pasta CH → index.html)
python onyx/onyx_web_preview.py \
    --input  resultados/charts/MinhaMusica/ \
    --output resultados/previews/MinhaMusica/

# ... ou em lote (todas as sub-pastas com notes.mid):
python onyx/onyx_web_preview.py \
    --input resultados/charts/ \
    --output resultados/previews/ --batch
```

---

## Formato `notes.xlsx` resumido

O formato de intercâmbio entre os modelos e o `excel_to_midi.py`:

**Aba `info`** — 1 cabeçalho + 1 linha:
```
File Name | MIDI Type | Ticks per Beat | Tempo (µs/beat) | BPM | Time Signature
```

**Abas `drums` / `guitar` / `rhythm` / `vocals`** — cabeçalho + N linhas:
```
# | Note # | Note Name | Channel | Velocity |
Start Tick | Start (s) | End Tick | End (s) | Duration (ticks) | Duration (s)
```

Mapeamento aba → track MIDI (em `excel_to_midi.py`):

| Aba    | Track MIDI    |
|--------|---------------|
| drums  | PART DRUMS    |
| guitar | PART GUITAR   |
| rhythm | PART BASS     |
| vocals | PART VOCALS   |

---

## Status

| Componente | Arquivo | Status |
|---|---|---|
| Separação de áudio | `processamento/audio/separa_audio.py` | ✅ Funciona |
| RB → CH em lote | `onyx/onyx_rb_to_ch.py` | ✅ Funciona |
| Organização do dataset | `dados/organizar_dataset.py` | ✅ Funciona |
| MIDI → Excel (detalhado) | `processamento/midi_excel/midi_to_excel.py` | ✅ Funciona |
| Excel → MIDI | `processamento/midi_excel/excel_to_midi.py` | ✅ Funciona |
| song.ini | `song_ini.py` | ✅ Funciona (diff_* fica em -1) |
| Web preview | `onyx/onyx_web_preview.py` | ✅ Funciona |
| Modelo drums | `treinamento/drum_crnn.py` + notebook | ✅ Funciona (em evolução) |
| Inferência drums | `treinamento/modelo_gera_excel.py` | ✅ Funciona |
| Modelo guitar | `treinamento/guitar_crnn.py` | 🔲 A criar |
| Modelo bass | `treinamento/bass_crnn.py` | 🔲 A criar |
| Modelo vocals | `treinamento/vocals_crnn.py` | 🔲 A criar |
| Orquestrador completo | `main.py` | 🔲 A criar (após os 4 modelos) |

---

## Convenções de código

- **Todos os scripts `.py`** têm dupla interface:
  função pública importável + CLI com `argparse`.
- **Nada de Colab/Drive** (`google.colab`, `drive.mount`, `files.upload`).
- **Caminhos** sempre relativos à raiz do repositório ou passados por
  argumento; nunca hardcoded dentro de funções.
- **Notebooks** (`.ipynb`) são usados apenas para treinamento. Todo o resto
  é `.py`.
- **Sem duplicação**: `onyx_web_preview.py` importa `resolve_onyx_binary`
  de `onyx_rb_to_ch.py`; `organizar_dataset.py` implementa internamente a
  conversão `mid→xlsx` (não depende de `midi_to_excel.py`, que produz o
  formato detalhado de análise).

---

## Modelo de drums — `DrumCRNN`

Arquitetura:

- **Entrada:** mel-espectrograma normalizado do `drums.ogg`, shape
  `[N_MELS=128, n_steps]`. Cada frame corresponde a 1 semicolcheia
  (`SUBDIV_PER_BEAT=4`).
- **CNN:** 3 blocos (32 → 64 → 128 filtros), pool 2 só no eixo de
  frequência.
- **BiLSTM:** 2 camadas, hidden=256.
- **Head:** MLP → 5 saídas (lanes Kick / Snare / Yellow / Blue / Green).

Lanes do drums (mapeamento MIDI Rock Band):

| Idx | MIDI | Lane |
|---|---|---|
| 0 | 24 (C1)  | Kick |
| 1 | 26 (D1)  | Snare |
| 2 | 27 (D#1) | Yellow (hi-hat) |
| 3 | 30 (F#1) | Blue (tom blue) |
| 4 | 31 (G1)  | Green (crash) |

Treino: BCE + `pos_weight` por lane (lanes raras), early stopping pelo
F1 macro de validação, threshold tuning por lane após o treino.

---

## Licença

A definir.
