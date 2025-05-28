# Base image
FROM python:3.6

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean

# Install Python packages
RUN pip install --upgrade pip && \
    pip install \
        numpy \
        pandas \
        scikit-learn \
        tensorflow==2.6.2 \
        shap \
        matplotlib \
        logomaker \
        imbalanced-learn

# Default command
CMD ["/bin/bash"]
