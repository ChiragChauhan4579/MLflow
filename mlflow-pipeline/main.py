import mlflow
import click

def _run(entrypoint, parameters={}, source_version=None, use_cache=True):
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return submitted_run

@click.command()
def workflow():
    with mlflow.start_run(run_name ="pipeline_example") as active_run:
        _run("load_data")
        _run("train_model")
        
if __name__=="__main__":
    workflow()