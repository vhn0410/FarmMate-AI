This is an API for RAG & Agent framework with an OOP code structure easy to modify agent architecture
Using Untructured to extract different type of files (pdf, docs, exel,...)  

# python main.py -> run RAG pipeline with system documents 
# uvicorn app:app --reload -> To start development API

# System Dependencies

To get started with Unstructured.io, we need a few system-wide dependencies: 

## Poppler (poppler-utils)
Handles PDF processing. It's a library that can extract text, images, and metadata from PDFs. Unstructured uses it to parse PDF documents and convert them into processable text.

install using choco (WINDOW): choco install poppler

## Tesseract (tesseract-ocr) 
Optical Character Recognition (OCR) engine. When you have scanned documents, images with text, or PDFs that are essentially pictures, Tesseract reads the text from these images and converts it to machine-readable text.

install using choco (WINDOW): choco install tesseract

## libmagic
File type detection library. It identifies what type of file you're dealing with (PDF, Word doc, image, etc.) by analyzing the file's content, not just the extension. This helps Unstructured choose the right processing method for each document.

pip install python-magic-bin

Python lib:
%pip install -Uq "unstructured[all-docs]" 
%pip install -Uq langchain_chroma 
%pip install -Uq langchain langchain-community langchain-openai 
%pip install -Uq python_dotenv


# Check support GPU 
python -c "import torch; print('='*80); print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No GPU detected'); print('='*80)"

# install requirements
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

# tesseract (vietnamese package)
https://github.com/tesseract-ocr/tessdata/blob/main/vie.traineddata

import pytesseract
print(pytesseract.get_languages(config=''))


# Sensorthings API
http://localhost:8026/get-things?filter=properties/user_id%20eq%20%27a2ecd084-3013-4a15-836c-9c0b7be0b320%27&expand=sensors($expand=datastreams($expand=observedproperties,measurementunits))


data:
[
    [
        {
            "id": 3,
            "name": "Handle device",
            "description": "Handle device",
            "properties": {
                "user_id": "a2ecd084-3013-4a15-836c-9c0b7be0b320"
            },
            "sensors": [
                {
                    "id": 4,
                    "name": "Soil Integrated Sensor",
                    "description": "Soil Integrated Sensor",
                    "encodingType": "application\/pdf",
                    "metadata": "Soil Integrated Sensor",
                    "datastreams": [
                        {
                            "id": 10,
                            "name": "Temperature",
                            "thingId": 3,
                            "sensorId": 4,
                            "description": "Temperature",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 4,
                                    "name": "Temperature",
                                    "definition": "Temperature",
                                    "description": "Temperature"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 1,
                                    "name": "degree Celsius",
                                    "symbol": "\u00b0C",
                                    "definition": "http:\/\/unitsofmeasure.org\/ucum.html#para-30"
                                }
                            ]
                        },
                        {
                            "id": 9,
                            "name": "Moisture",
                            "thingId": 3,
                            "sensorId": 4,
                            "description": "Moisture",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 3,
                                    "name": "Moisture",
                                    "definition": "Moisture",
                                    "description": "Moisture"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 2,
                                    "name": "percent",
                                    "symbol": "%",
                                    "definition": "http:\/\/unitsofmeasure.org\/ucum.html#para-29"
                                }
                            ]
                        },
                        {
                            "id": 8,
                            "name": "EC",
                            "thingId": 3,
                            "sensorId": 4,
                            "description": "EC",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 9,
                                    "name": "EC",
                                    "definition": "EC",
                                    "description": "EC"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 5,
                                    "name": "\u00b5S\/cm",
                                    "symbol": "\u00b5S\/cm",
                                    "definition": "\u00b5S\/cm"
                                }
                            ]
                        },
                        {
                            "id": 7,
                            "name": "Potassium",
                            "thingId": 3,
                            "sensorId": 4,
                            "description": "Potassium",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 8,
                                    "name": "Potassium",
                                    "definition": "Potassium",
                                    "description": "Potassium"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 4,
                                    "name": "mg\/L",
                                    "symbol": "mg\/L",
                                    "definition": "mg\/L"
                                }
                            ]
                        },
                        {
                            "id": 6,
                            "name": "Phosphorus",
                            "thingId": 3,
                            "sensorId": 4,
                            "description": "Phosphorus",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 7,
                                    "name": "Phosphorus",
                                    "definition": "Phosphorus",
                                    "description": "Phosphorus"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 4,
                                    "name": "mg\/L",
                                    "symbol": "mg\/L",
                                    "definition": "mg\/L"
                                }
                            ]
                        },
                        {
                            "id": 5,
                            "name": "Nitrogen",
                            "thingId": 3,
                            "sensorId": 4,
                            "description": "Nitrogen",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 6,
                                    "name": "Nitrogen",
                                    "definition": "Nito",
                                    "description": "Nito"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 4,
                                    "name": "mg\/L",
                                    "symbol": "mg\/L",
                                    "definition": "mg\/L"
                                }
                            ]
                        },
                        {
                            "id": 4,
                            "name": "pH",
                            "thingId": 3,
                            "sensorId": 4,
                            "description": "pH",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 5,
                                    "name": "pH",
                                    "definition": "pH",
                                    "description": "pH"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 3,
                                    "name": "H+",
                                    "symbol": "H+",
                                    "definition": "H+"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "id": 2,
            "name": "Thi\u1ebft b\u1ecb \u0111o th\u00f4ng s\u1ed1 \u0111\u1ea5t",
            "description": "Thi\u1ebft b\u1ecb \u0111o th\u00f4ng s\u1ed1 \u0111\u1ea5t",
            "properties": {
                "user_id": "a2ecd084-3013-4a15-836c-9c0b7be0b320"
            },
            "sensors": [
                {
                    "id": 3,
                    "name": "ES-SM-TH-01",
                    "description": "C\u1ea3m bi\u1ebfn \u0111\u1ed9 \u1ea9m \u0111\u1ea5t, nhi\u1ec7t \u0111\u1ed9 \u0111\u1ea5t ES-SM-TH-01 ( RS485 | 4-20mA | 0-10V)",
                    "encodingType": "application\/pdf",
                    "metadata": "https:\/\/epcb.vn\/products\/cam-bien-do-do-am-nhiet-do-dat-es-sm-th-01?gidzl=xB-4VH1JDJFleu1z4KKpDk-kWduAHpPZ-Ao7U0eMD3JqgjP-3H5XDANwtIjP4Jimzws6A3VBteHv5L0sCW",
                    "datastreams": [
                        {
                            "id": 3,
                            "name": "Nhi\u1ec7t \u0111\u1ed9 \u0111\u1ea5t",
                            "thingId": 2,
                            "sensorId": 3,
                            "description": "Nhi\u1ec7t \u0111\u1ed9 \u0111\u1ea5t",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 4,
                                    "name": "Temperature",
                                    "definition": "Temperature",
                                    "description": "Temperature"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 1,
                                    "name": "degree Celsius",
                                    "symbol": "\u00b0C",
                                    "definition": "http:\/\/unitsofmeasure.org\/ucum.html#para-30"
                                }
                            ]
                        },
                        {
                            "id": 2,
                            "name": "\u0110\u1ed9 \u1ea9m \u0111\u1ea5t",
                            "thingId": 2,
                            "sensorId": 3,
                            "description": "\u0110\u1ed9 \u1ea9m \u0111\u1ea5t",
                            "observationType": 4,
                            "observedproperties": [
                                {
                                    "id": 3,
                                    "name": "Moisture",
                                    "definition": "Moisture",
                                    "description": "Moisture"
                                }
                            ],
                            "measurementunits": [
                                {
                                    "id": 2,
                                    "name": "percent",
                                    "symbol": "%",
                                    "definition": "http:\/\/unitsofmeasure.org\/ucum.html#para-29"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
]