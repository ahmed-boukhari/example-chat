
import {
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteria,
    env
} from '@huggingface/transformers';

env.localModelPath = '/Users/ahhmed/Downloads/transformers.js-main/examples/webgpu-chat/models/';
env.allowRemoteModels = false;
env.allowLocalModels = true;

//https://github.com/microsoft/onnxruntime/issues/24325
//https://github.com/huggingface/transformers.js/?tab=readme-ov-file
env.backends.onnx.wasm.wasmPaths = '/Users/ahhmed/Downloads/transformers.js-main/examples/webgpu-chat/wasm/';

class CallbackTextStreamer extends TextStreamer {
    constructor(tokenizer, cb) {
        super(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
        });
        this.cb = cb;
    }

    on_finalized_text(text) {
        this.cb(text);
    }
}

class InterruptableStoppingCriteria extends StoppingCriteria {
    constructor() {
        super();
        this.interrupted = false;
    }

    interrupt() {
        this.interrupted = true;
    }

    reset() {
        this.interrupted = false;
    }

    _call(input_ids, scores) {
        return new Array(input_ids.length).fill(this.interrupted);
    }
}

const stopping_criteria = new InterruptableStoppingCriteria();

async function hasFp16() {
    // try {
    //     const adapter = await navigator.gpu.requestAdapter();
    //     return adapter.features.has('shader-f16');
    // } catch (e) {
        return false;
    // }
}

/**
 * This class uses the Singleton pattern to ensure that only one instance of the model is loaded.
 */
class TextGenerationPipeline {
    static model = null;
    static tokenizer = null;
    static streamer = null;
    static model_id = null;
    static dtype = null;

    static async getInstance(progress_callback = null) {
        // Choose the model based on whether fp16 is available
         this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
            progress_callback,
        });
        this.tokenizer.then(t => console.log('tokenizer:', t.config.tokenizer_class));

        this.model ??= AutoModelForCausalLM.from_pretrained(this.model_id, {
            dtype: this.dtype,
            device: "auto",
            progress_callback,
        });

        return Promise.all([this.tokenizer, this.model]);
    }
}

async function generate(messages) {
    // Retrieve the text-generation pipeline.
    const [tokenizer, model] = await TextGenerationPipeline.getInstance();

    const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
    });

    let startTime;
    let numTokens = 0;
    const cb = (output) => {
        startTime ??= performance.now();

        let tps;
        if (numTokens++ > 0) {
            tps = numTokens / (performance.now() - startTime) * 1000;
        }
        self.postMessage({
            status: 'update',
            output, tps, numTokens,
        });
    }

    const streamer = new CallbackTextStreamer(tokenizer, cb);

    // Tell the main thread we are starting
    self.postMessage({ status: 'start' });

    const outputs = await model.generate({
        ...inputs,
        max_new_tokens: 512,
        streamer,
        stopping_criteria,
    });
    const outputText = tokenizer.batch_decode(outputs, { skip_special_tokens: false });

    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: outputText,
    });
}

async function load(model_id, dtype) {
    self.postMessage({
        status: 'loading',
        data: 'Loading model...'
    });
    TextGenerationPipeline.model_id ??= model_id;
    TextGenerationPipeline.dtype ??= dtype;
    // Load the pipeline and save it for future use.
    const [tokenizer, model] = await TextGenerationPipeline.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    });

    self.postMessage({
        status: 'loading',
        data: 'Compiling shaders and warming up model...'
    });

    // Run model with dummy input to compile shaders
    const inputs = tokenizer('a');
    await model.generate({ ...inputs, max_new_tokens: 1 });
    self.postMessage({ status: 'ready' });
}
// Listen for messages from the main thread
self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'load':
            load(data.model_id, data.dtype);
            break;

        case 'generate':
            stopping_criteria.reset();
            generate(data);
            break;

        case 'interrupt':
            stopping_criteria.interrupt();
            break;

        case 'reset':
            stopping_criteria.reset();
            break;
    }
});
