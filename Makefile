.PHONY: data up down clean-attack flip chaff ood logs

data:
	pip install numpy pandas pyarrow --quiet
	DATA_OUT=./data_out python data/prepare_nslkdd.py

up:
	docker compose up --build

down:
	docker compose down -v

clean-attack:
	ATTACK_MODE=clean docker compose up -d simulator
flip:
	ATTACK_MODE=label_flip POISON_FRAC=0.3 docker compose up -d simulator
chaff:
	ATTACK_MODE=chaff docker compose up -d simulator
ood:
	ATTACK_MODE=ood docker compose up -d simulator

logs:
	docker compose logs -f gate simulator training
