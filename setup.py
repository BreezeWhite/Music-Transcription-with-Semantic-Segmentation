from setuptools import setup, find_packages

setup(
    name='Music-Transcription-with-Semantic-Segmentation',
    version='1.0.0',
    description='A research project about Automatic Music Transcription (AMT) task',
    author='MCTLab, IIS, Academia Sinica',
    author_email='freedombluewater@gmail.com',
    packages=find_packages(),
    install_requires=[
        'keras>=2.2.4',
        'tensorflow-gpu==1.15.0',
        'tensorflow-probability==0.6',
        'tensor2tensor',
        'librosa',
        'pretty_midi'
    ]
)