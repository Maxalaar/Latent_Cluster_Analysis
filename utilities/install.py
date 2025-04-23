import subprocess

def install_command(command):
    try:
        print('-- --')
        print(f'Running command: {command}')
        subprocess.run(command, shell=True, check=True)
        print(f'Successfully ran command: {command}')
        print()
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while running command: {command}. Error: {str(error)}')
    except FileNotFoundError as fnf_error:
        print(f'Command not found: {command}. Error: {str(fnf_error)}')

if __name__ == '__main__':
    commands = [
        'pip install torch torchvision torchaudio',
        'pip install lightning',
        'pip install -U tensorboard',
        'pip install requests',
        'pip install scikit-learn',
        'pip install matplotlib',
        'pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==24.10.*" "dask-cudf-cu12==24.10.*" "cuml-cu12==24.10.*" "cugraph-cu12==24.10.*" "nx-cugraph-cu12==24.10.*" "cuspatial-cu12==24.10.*" "cuproj-cu12==24.10.*" "cuxfilter-cu12==24.10.*" "cucim-cu12==24.10.*" "pylibraft-cu12==24.10.*" "raft-dask-cu12==24.10.*" "cuvs-cu12==24.10.*" "nx-cugraph-cu12==24.10.*"',
        'pip install hdbscan'
    ]

    # Execute conda commands
    for command in commands:
        install_command(command)
