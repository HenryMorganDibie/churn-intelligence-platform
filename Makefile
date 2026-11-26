IMAGE ?= churn-intelligence:local
CLUSTER ?= churn

.PHONY: install test train mlflow-ui image kind-up kind-load deploy-dev deploy-mlflow retrain smoke kind-down

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

train:                 ## train locally, log to ./mlruns, apply the promotion gate
	python -m app.models.trainer

mlflow-ui:             ## needs the full mlflow package: pip install mlflow==2.17.2
	mlflow ui --backend-store-uri ./mlruns --port 5000

image:
	docker build -t $(IMAGE) .

kind-up:
	kind create cluster --name $(CLUSTER)

kind-load: image
	kind load docker-image $(IMAGE) --name $(CLUSTER)

deploy-dev: kind-load  ## API only, model baked into the image
	kubectl apply -k deploy/k8s/overlays/dev
	kubectl -n churn rollout status deploy/churn-api --timeout=300s

deploy-mlflow: kind-load  ## MLflow server + training Job + API serving the champion alias
	kubectl apply -k deploy/k8s/overlays/mlflow
	kubectl -n churn rollout status deploy/mlflow --timeout=300s
	kubectl -n churn wait --for=condition=complete job/churn-train --timeout=600s
	kubectl -n churn rollout status deploy/churn-api --timeout=300s

retrain:               ## run the training Job again; the gate decides whether it becomes champion
	kubectl -n churn delete job churn-train --ignore-not-found
	kubectl apply -k deploy/k8s/overlays/mlflow
	kubectl -n churn wait --for=condition=complete job/churn-train --timeout=600s
	kubectl -n churn logs job/churn-train | tail -3
	kubectl -n churn rollout restart deploy/churn-api

smoke:
	kubectl -n churn port-forward svc/churn-api 8080:80 & sleep 4; \
	curl -fsS localhost:8080/health/ready && echo && curl -fsS localhost:8080/health/model && echo; \
	kill %1

kind-down:
	kind delete cluster --name $(CLUSTER)
