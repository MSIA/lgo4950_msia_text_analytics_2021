train_models:
	python run_logit.py

topic_models:
	python run_topic_model.py

app_image:
	docker build -f app/Dockerfile_app -t app_lcg:latest .

run_app:
	docker run -d -p 5000:5000 --name app_instance app_lcg

kill-app:
	docker kill app_instance