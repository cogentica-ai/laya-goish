# Laya Goish

## Turn text into decisions.

Route a support ticket. Score urgency. Detect a refund request. Laya Goish turns English and Thai text into structured answers your application can use, running on your own infrastructure.

[Download v0.1.0](https://github.com/cogentica-ai/laya-goish/releases/tag/v0.1.0) · [Get started](docs/getting-started.md) · [API reference](docs/http.md)

### Ask for the answer your workflow needs

| Task | What you get | Put it to work |
| --- | --- | --- |
| **Choose** | A choice from your options, with probabilities and confidence | Route tickets, classify messages, select a next step |
| **Score** | A score against your ordered criteria | Assess urgency, sentiment, or quality |
| **Check** | The probability that a condition is true | Detect intent, flag a request, check a requirement |

Define the questions and criteria. Send text or a JSON object. Receive structured JSON.

```sh
curl http://localhost:8080/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laya",
    "state": "I was charged twice. Please refund the duplicate payment.",
    "questions": {
      "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?"
      }
    }
  }'
```

[Explore example requests](examples) · [Use built-in presets](docs/getting-started.md#use-a-preset)

### English and Thai. One API.

Use **Laya** for English decisions or **OpenThai-SystemOne** for Thai and English. Load both into one server and select a model in each request.

| Model | Languages | Guide |
| --- | --- | --- |
| Laya English | English | [Download and run](docs/getting-started.md) |
| OpenThai-SystemOne v0.3 | Thai and English | [Convert and run](docs/openthai.md) |

[Serve multiple models](docs/models.md)

### Run where your data lives

A single Linux executable runs inference on your CPU. No hosted inference service is required. Keep the models on your machine and integrate through HTTP, the command line, or a persistent JSON-lines process.

Batch requests let you apply the same questions to a collection of inputs. Optional bearer authentication controls access to the decision API.

### Start with Laya

Download and extract the [Linux release](https://github.com/cogentica-ai/laya-goish/releases/tag/v0.1.0), then run:

```sh
bash scripts/download-models.sh f16
./laya serve models/laya_english_f16.gguf --port 8080 --threads 4
```

Requires Linux x86-64 with AVX2/FMA. Model weights are downloaded separately.

[Installation and requirements](docs/getting-started.md) · [HTTP API](docs/http.md) · [Release notes](CHANGELOG.md)

---

Built with Goish Rust. Laya inference is ported from [ggmlc](https://github.com/monatis/ggmlc); Thai/English support uses [OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne).

[Architecture](docs/implementation.md) · [Validation](docs/validation.md) · [Attribution](NOTICE)
