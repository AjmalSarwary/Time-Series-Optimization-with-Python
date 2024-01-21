{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO1uHI+hLQXdhzkrwCXHI9Y",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AjmalSarwary/invest_ml/blob/master/code/limits_of_diversification.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-zbbUVGuyH37",
        "outputId": "6c3457c4-1563-4d67-d148-43b3091aab1c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'invest_ml'...\n",
            "remote: Enumerating objects: 173, done.\u001b[K\n",
            "remote: Counting objects: 100% (173/173), done.\u001b[K\n",
            "remote: Compressing objects: 100% (128/128), done.\u001b[K\n",
            "remote: Total 173 (delta 48), reused 119 (delta 34), pack-reused 0\u001b[K\n",
            "Receiving objects: 100% (173/173), 1.49 MiB | 7.53 MiB/s, done.\n",
            "Resolving deltas: 100% (48/48), done.\n",
            "/content/invest_ml\n"
          ]
        }
      ],
      "source": [
        "!git clone https://github.com/ajmalsarwary/invest_ml.git\n",
        "%cd /content/invest_ml\n",
        "import sys\n",
        "sys.path.append('/content/invest_ml/code')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import invest_risk_kit as rk\n",
        "import pandas as pd\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "59KYc0jOyK2g"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ind_retrun = rk.get_ind_returns()"
      ],
      "metadata": {
        "id": "5Nz8jpO2yUvU"
      },
      "execution_count": 3,
      "outputs": []
    }
  ]
}