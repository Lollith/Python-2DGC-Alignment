# Python-2DGC


<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Easy47/Python-2DGC/">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Python-2DGC</h3>

  <p align="center">
    Python GCxGC-MS data processing
    <br />
    ·
    <a href="https://github.com/Lollith/Python-2DGC-Alignment/issues">Report Bug</a>
    ·
    <a href="https://github.com/Lollith/Python-2DGC-Alignment/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#credits">Credits</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#usage-examples">Usage Examples</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Python package to process GCxGC-MS data.

* Processing and annotation of chromatograms
* Chromatogram visualization
* Peak tables alignment and chromatogram alignment using R package
* Machine learning and statistical analysis for the automatic detection of biomarkers
* Pixel discriminant approach for the automatic detection of biomarkers
* GCxGC-MS data simulation


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)][Python-url]
* [![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)][Jupyter-url]
* [![R](https://img.shields.io/badge/r-%23276DC3.svg?style=for-the-badge&logo=r&logoColor=white)][R-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Credits

This project is basedon the work of [Nicolas Romano]( https://github.com/Easy47/Python-2DGC) & [Ahmed Hassayoune](https://github.com/ahmedhassayoune/Python-2DGC-Alignment).

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
#### Volumes

Create a folder on your host machine that will be mounted into the Docker container.
For easier usage, it is recommended to use a path that ends with .../app/data, so the volume is properly mapped inside the container.

#### .env

Create a .env file at the root of the project with the following variables:
```
# Password hash for JupyterLab login (can be generated with `IPython.lib.passwd()`)
JUPYTER_PASSWORD_HASH= ...

# Absolute path to the folder on your host machine that will be mounted into the Docker container
HOST_VOLUME_PATH= .../app/data

# Path inside the container where data will be available
DOCKER_VOLUME_PATH=/app/data 
 
```

### Installation

1. Clone the repo
```bash
git clone https://github.com/Lollith/Python-2DGC-Alignment.git
```

2. Open a terminal in the project directory and run:
```bash
make
```

### NIST Integration

In this project, we integrated the NIST mass spectral search engine using Docker to automate the identification of chemical compounds. 
We could not directly call the NIST Docker container from inside our own container. To resolve this, we took the following approach:
- Custom Docker Container for NIST:
    Instead of relying solely on the pre-built NIST image from Docker Hub, we built our own Docker container based on domdfcoding/pywine-pyms-nist. This allowed us to mount our own NIST library (mainlib) and temporary directory.
- Service Communication via Docker Compose:
    The app can call NIST using http://nist:5001
- Keeping NIST Persistent:
    We can rebuild our app without affecting the NIST container.

To use NIST, you need to place your NIST database files in the `..../..../volume_data/` directory.


### Make Commands

The project uses a Makefile to simplify common operations. Here are the main commands:

- `make` : Checks if the Docker image is built, builds it if necessary, then starts the container
- `make stop` : Stops all containers
- `make clean` : Cleans Docker images and volumes
- `make re` : Restarts the application (cleans and rebuilds)
- `make re_dev` : Cleans, rebuild and starts the development environment
- `make logs_dev` : Shows development environment logs


### Usage

The project uses a Jupyter notebook interface for parameter selection and analysis:

1. Open your browser and navigate to http://localhost:8888

2. Enter password (instead of copying the Jupyter URL with token, you now just enter a password)

3. In the Jupyter interface:
   - Choose the paths and files to analyze
   - Select your analysis parameters
   - Run the `sample_identification` function to process your GCxGC-MS data

The analysis will be performed using the selected parameters and the results will be available in the Jupyter notebook.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage Examples

Here are some examples of how to use the tool:

1. **Process a single chromatogram file:**
   - Select the file path in the Jupyter interface
   - Choose your analysis parameters
   - Run the analysis

2. **Process multiple files in a cohort:**
   - Select the directory containing all your chromatogram files
   - To analyze all files in the directory, leave the file field empty
   - Choose your analysis parameters
   - Run the analysis to process all files

3. **Customize analysis parameters:**
   - Adjust peak detection thresholds
   - Modify identification criteria
   - Set alignment parameters

_For an **overview of the functions**, read the **detailed documentation of a specific function directly in its file**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

* [Adeline Gouet](mailto:adeline.gouet@gmail.com)

* [Camille Roquencourt](mailto:camille.roquencourt@hotmail.fr)

* [Nicolas Romano](mailto:nicolas.romano@epita.fr)


Project Link: [https://github.com/Lollith/Python-2DGC-Alignment](https://github.com/Lollith/Python-2DGC-Alignment)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white

[Python-url]: https://python.org/
[Jupyter-url]: https://jupyter.org/
[R-url]: https://www.r-project.org/
