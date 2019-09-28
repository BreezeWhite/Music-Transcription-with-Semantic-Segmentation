from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='caffinipa',
    version='1.0.0',
    description='A research project about Automatic Music Transcription (AMT) task',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='MCTLab, IIS, Academia Sinica',
    author_email='freedombluewater@gmail.com',
    packages=find_packages(),
    install_requires=[
        'keras>=2.2.4',
        'tensorflow-gpu==1.13.1',
        'tensorflow-probability==0.6',
        'PyYAML>=5.1',
        'tensor2tensor',
        'librosa',
        'pretty_midi'
    ]
)
