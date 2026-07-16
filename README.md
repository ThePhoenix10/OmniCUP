# OmniCUP
A biologically interpretable hybrid ensemble framework for primary-site prediction in Cancer of Unknown Primary (CUP) using routine-care next-generation sequencing data.

## Overview
OmniCUP is a hybrid ensemble machine learning framework designed to infer the most likely tissue of origin for cancers of unknown primary (CUP) using routinely collected next-generation sequencing (NGS) data.

Combining 10 neural networks with 10 XGBoost models, the framework has been externally validated using an independent precision oncology cohort and deployed as a real-time, interpretable web application.

## Paper
[Read the paper](https://drive.google.com/file/d/1LfIgeSVCdTYQ-2yEE7zrdYnzlhq3BiOC/view?usp=sharing)

**OmniCUP: A Biologically Interpretable Hybrid Ensemble Framework for Primary-Site Prediction in Cancer of Unknown Primary Using Routine-Care Next-Generation Sequencing Data**
Saicharan Vellanki (Issaquah High School), Paraic Kenny (Gundersen Medical Foundation)

## Web Application
The live web application can be accessed at: https://omnicup.live/

Every prediction is accompanied by a per-feature contribution breakdown, showing how much each feature contributed to each candidate tumor type and whether its effect was positive or negative — reducing the black-box nature of the model for clinicians and researchers.

## Data

### Training Data
* MSK-MetTropism cohort (25,755 de-identified tumor samples, 27 cancer types), publicly available via cBioPortal: [msk_met_2021](https://www.cbioportal.org/study/summary?id=msk_met_2021)

### External Validation
* Gundersen Precision Oncology Cohort (19-sample validation subset, from a broader 770-sample, 54-tissue-origin cohort)

## Installation

### Clone the Repository
```
git clone https://github.com/ThePhoenix10/OmniCUP.git
cd OmniCUP
```

### System Requirements
* Python 3.8+
* pip
* (Optional) VS Code or other IDE

### Create Virtual Environment (Recommended)
```
python -m venv venv
```

Activate:
Mac/Linux:
```
source venv/bin/activate
```
Windows:
```
venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

## Authors

* Saicharan Vellanki — Issaquah High School
* Paraic Kenny, PhD — Gundersen Medical Foundation

## Contact

* [saivellanki10@gmail.com](mailto:saivellanki10@gmail.com)
* [pakenny@emplifyhealth.org](mailto:pakenny@emplifyhealth.org)

## Acknowledgment
The authors thank the Gundersen Medical Foundation for its collaboration on external validation and for providing access to the Precision Oncology Cohort used in this study.

## License

```
MIT License

Copyright (c) 2026 Saicharan Vellanki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
