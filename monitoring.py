import os
import time
import webbrowser
from multiprocessing import Process
from typing import List
import requests


def run_tensorboard():
    path = os.getcwd()
    print(path)
    os.system('tensorboard --bind_all --logdir ' + str(path) + '/experiments/')


def list_to_url(list_monitoring: List[str]) -> str:
    url = 'http://localhost:6006/?pinnedCards=['

    for element in list_monitoring:
        url += '{"plugin"%3A"scalars"%2C"tag"%3A"'
        url += element.replace('/', '%2F')
        url += '"}%2C'

    url = url.rstrip('%2C')  # Remove the last comma
    url += ']&darkMode=true#timeseries'

    # Encode special characters
    url = url.replace('[', '%5B').replace(']', '%5D')
    url = url.replace('{', '%7B').replace('}', '%7D')
    url = url.replace('"', '%22')

    return url


def wait_for_tensorboard(host: str = 'http://localhost', port: int = 6006, timeout: int = 60):
    """Waits for TensorBoard to start by checking its availability on the specified host and port."""
    url = f"{host}:{port}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)  # Retry every 0.5 seconds
    raise TimeoutError(f"TensorBoard did not start within {timeout} seconds.")


if __name__ == "__main__":
    list = [
        'train_loss_epoch',
        'validation_loss_epoch',
    ]

    process = Process(target=run_tensorboard)
    process.start()

    try:
        # Wait for TensorBoard to be ready
        wait_for_tensorboard()
    except TimeoutError as e:
        print(e)
    else:
        # Open monitoring dashboards
        webbrowser.open_new('http://127.0.0.1:8265')
        webbrowser.open(list_to_url(list))
