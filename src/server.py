import aiohttp
import asyncio

import fastai
import uvicorn

from fastai.text import *
from fastai.text.data import get_default_size, full_char_coverage_langs
from fastai.core import PathOrStr
from sklearn.metrics import f1_score
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse


url_classifier = 'https://www.dropbox.com/s/9nnbpija7uiha4o/es_clas_intervenciones_congreso_sp_multifit.pth?raw=1'
filename_classifier = 'es_clas_intervenciones_congreso_sp_multifit'
url_data_clas = 'https://www.dropbox.com/s/o1b3u1iz5x7h2xl/es_textlist_class_intervenciones_congreso_sp_multifit?raw=1'
filename_data_clas = 'es_textlist_class_intervenciones_congreso_sp_multifit'
url_spm_model = 'https://www.dropbox.com/s/62kkzryul69hjxv/spm_fwd.model?raw=1'
filename_sp_model = 'spm_fwd.model'
url_spm_vocab = 'https://www.dropbox.com/s/j3dq9gvmbw36xy7/spm_fwd.vocab?raw=1'
filename_sp_vocab = 'spm_fwd.vocab'

classes = ['GCUP-EC-GC', 'GCs', 'GP', 'GS', 'GVOX']

path = Path(__file__).parent
models_dir = path / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "on'


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])


@np_func
def f1(inp, targ):
    return f1_score(targ, np.argmax(inp, axis=-1), average='weighted')


def train_sentencepiece(texts: Collection[str], path: PathOrStr, vocab_sz: int = None, max_vocab_sz: int = 30000,
                        model_type: str = 'unigram', max_sentence_len: int = 20480, lang='en', char_coverage=None,
                        tmp_dir='tmp', enc='utf8'):
    """Train a sentencepiece tokenizer on `texts` and save it in `path/tmp_dir`"""
    from sentencepiece import SentencePieceTrainer
    cache_dir = Path(path) / tmp_dir
    os.makedirs(cache_dir, exist_ok=True)
    if vocab_sz is None:
        vocab_sz = get_default_size(texts, max_vocab_sz)
    raw_text_path = cache_dir / 'all_text.out'
    with open(raw_text_path, 'w', encoding=enc) as f:
        f.write("\n".join(texts))
    spec_tokens = ['\u2581' + s for s in defaults.text_spec_tok]
    SentencePieceTrainer.Train(" ".join([
        f"--input={raw_text_path} --max_sentence_length={max_sentence_len}",
        f"--character_coverage={ifnone(char_coverage, 0.99999 if lang in full_char_coverage_langs else 0.9998)}",
        f"--unk_id={len(defaults.text_spec_tok)} --pad_id=-1 --bos_id=-1 --eos_id=-1",
        f"--user_defined_symbols={','.join(spec_tokens)}",
        f"--model_prefix={cache_dir / 'spm'} --vocab_size={vocab_sz} --model_type={model_type}"]))
    raw_text_path.unlink()
    return cache_dir


fastai.text.data.train_sentencepiece = train_sentencepiece


def fix_sp_processor(learner: Learner) -> None:
    """
    Fixes SentencePiece paths serialized into the model.
    Parameters
    ----------
    learner
        Learner object
    """
    for processor in learner.data.processor:
        if isinstance(processor, SPProcessor):
            processor.sp_model = f'{models_dir}/spm_fwd.model'
            processor.sp_vocab = f'{models_dir}/spm_fwd.vocab'


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(url_classifier, models_dir / f'{filename_classifier}.pth')
    await download_file(url_data_clas, models_dir / f'{filename_data_clas}')
    await download_file(url_spm_model, models_dir / f'{filename_sp_model}')
    await download_file(url_spm_vocab, models_dir / f'{filename_sp_vocab}')
    try:
        bs = 18

        config = awd_lstm_clas_config.copy()
        config['qrnn'] = True
        config['n_hid'] = 1550  # default 1152
        config['n_layers'] = 4  # default 3

        data_clas = load_data(models_dir, filename_data_clas, bs=bs, num_workers=1)
        learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, drop_mult=0.3,
                                          metrics=[accuracy, f1])
        learn_c.path = path
        learn_c.load(filename_classifier, purge=False)
        fix_sp_processor(learner=learn_c)

        return learn_c
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU " \
                      "environment.\n\nPlease update the fastai library in your training environment and export your " \
                      "model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai. "
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learner = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/predict', methods=['POST'])
async def predict(request):
    data = await request.body()

    input_text = json.loads(data.decode('utf-8'))['input_text']
    pred = learner.predict(input_text)[2] * 100

    return JSONResponse({
        'Unidas Podemos': f'{pred[0]:.2f}%',
        'PSOE': f'{pred[3]:.2f}%',
        'Ciudadanos': f'{pred[1]:.2f}%',
        'PP': f'{pred[2]:.2f}%',
        'VOX': f'{pred[4]:.2f}%',
    })


@app.route('/', methods=['GET'])
def status(request):
    return JSONResponse(dict(status='OK'))


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8501, log_level="info")
