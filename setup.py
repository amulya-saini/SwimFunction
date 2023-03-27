import setuptools

with open('README.md', 'r') as fh:
  long_description = fh.read()

setuptools.setup(
    name='swimfunction',
    version='0.0.1',
    author='Nicholas Jensen',
    author_email='nick.jensen@wustl.edu',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MokalledLab/SwimFunction',
    project_urls = {
        'Bug Tracker': 'https://github.com/MokalledLab/SwimFunction'
    },
    license='MIT',
    packages=['swimfunction'],
    install_requires=[],
)