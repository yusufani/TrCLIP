<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/trclip_arc.png" alt="Logo" width="1080" height="400">
  </a>

<h3 align="center">A General Purpose Turkish CLIP Model (TrCLIP) for Image&Text Retrieval and its Application to E-Commerce</h3>

  <p align="center">
    A turkish supported CLIP
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Paper(will be added)»</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo(will be added)</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
</div>






<!-- ABOUT THE PROJECT -->
## About The Project

In this paper, we introduce a Turkish adaption of CLIP (Contrastive Language-Image Pre-Training). Our approach is to train a model with the same output space as the Text encoder of the CLIP model while processing Turkish input. For this, we collected 2.5M unique English-Turkish data. The model we named TrCLIP performed  71\% in CIFAR100, 86\% in VOC2007, and 47\%  in FER2013 as zero-shot accuracy. We have examined its performance on e-commerce data and a vast domain-independent dataset in image and text retrieval tasks. The model can work in Turkish without any extra fine-tuning. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Example outputs -->
## Text retrieval
![image info](./images/text_ret.png)
![image info](./images/text_ret_sing.png)



## Image retrieval
![image info](./images/image_ret.jpg)
![image info](./images/image_ret_sing.png)



### Prerequisites

See requriments.txt

### Installation

! pip install trclip


<!-- USAGE EXAMPLES -->

Model can be found in huggingface : https://huggingface.co/yusufani/trclip-vitl14-e10

Detailed usage will be added.

Model space will be added.







<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - yusufani8@gmail.com

Project Link: [https://github.com/yusufani/TrCLIP](https://github.com/yusufani/TrCLIP)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




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
[linkedin-url]: https://linkedin.com/in/yusufani
[product-screenshot]: images/screenshot.png
