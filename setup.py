from setuptools import setup, find_packages

setup(
    name='emg-gesture-decoder',
    version='1.0.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.2.0',
        'torch>=2.0.0',
        'lightgbm>=4.0.0',
        'matplotlib>=3.7.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
    ],
    author='Yugandhar Kulkarni',
    description='EMG-Based Hand Gesture Decoding - Ninapro DB2 (4-Model Comparison)',
)
