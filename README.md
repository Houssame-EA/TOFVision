# SP-TOF-ICP-MS Analysis 'TOFVision'

A Streamlit web application TOFVision for analyzing Time-of-Flight Single Particle Inductively Coupled Plasma Mass Spectrometry (SP-TOF-ICP-MS) data, developed at the Prof. Kevin J. Wilkinson Laboratory, Université de Montréal.


## Features

- **Single File Analysis**
  - Mass distribution analysis
  - Element distribution visualization
  - Heatmap generation

- **Multi-File Analysis**
  - Batch processing of multiple files
  - Comparative analysis across samples

- **Isotopic Ratio Analysis**
  - Mole ratio analysis
  - Isotopic ratio calculations
  - Natural abundance comparisons

## Installation

1. Ensure you have Python 3.8 or newer installed
2. Clone the repository:
```bash
git clone https://github.com/Houssame-EA/TOFVision.git
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run TOFVision.py
```

2. Access the application through your web browser (typically http://localhost:8501)

3. Choose analysis type:
   - Single File Analysis: Upload and analyze individual SP-TOF-ICP-MS data files (Nu Quant Vitesse and SPCal format)
   - Multi Files Analysis: Process and compare multiple files simultaneously (Nu Quant Vitesse)
   - Isotopic Ratio Analysis: Perform detailed isotopic ratio calculations (Nu Quant Vitesse)

4. Follow the sidebar options to customize your analysis

## Data Format

The application accepts the following file formats:
- CSV (.csv)

Data should be structured in the following formats:
- NuQuant format
- SPCal format

## Dependencies

- streamlit>=1.28.0
- pandas>=2.0.0
- plotly>=5.18.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- scipy>=1.10.0
- statsmodels>=0.14.0
- mendeleev>=0.15.0
- openpyxl>=3.1.2

## Examples

![example1](https://github.com/user-attachments/assets/d44c135b-4aa6-4e48-8c50-0ef930528103)


## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Authors

- H-E Ahabchane
- Amanda Wu

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Citation

If you use this software in your research, please cite:
```
Ahabchane, H.-E.; Wu, A.; Wilkinson, K.J. SP-TOF-ICP-MS Analysis 'TOFVision'. 2024. https://github.com/Houssame-EA/TOFVision/tree/main

Université de Montréal, Prof. Kevin J. Wilkinson Laboratory.
```

## Acknowledgments

- Prof. Kevin J. Wilkinson Laboratory
- Département de Chimie
- Université de Montréal

## Support

For support, please contact [houssame-eddine.ahabchane@umontreal.ca] or visit our [Group website](https://kevinjwilkinson.openum.ca).
