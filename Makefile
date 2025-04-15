DEV_COMPOSE=docker-compose.dev.yml
PROD_COMPOSE=docker-compose.prod.yml

all_dev: build_dev start_dev
all_prod: build_prod start_prod

all: all_prod

build_dev: 
	docker compose -f $(DEV_COMPOSE) build

build_prod: 
	docker compose -f $(PROD_COMPOSE) build

start_dev :
	docker compose -f $(DEV_COMPOSE) up

start_prod :
	docker compose -f $(PROD_COMPOSE) up

stop:
	- docker compose -f $(DEV_COMPOSE) down
	-docker compose -f $(PROD_COMPOSE) down

logs_dev:
	docker compose -f $(DEV_COMPOSE) logs -f

logs_prod:
	docker compose -f $(PROD_COMPOSE) logs -f


clean: stop
		sudo docker system prune -af --volumes

re_dev: clean
	make all_dev

re: clean
	make all

.PHONY: all all_dev all_prod build_dev build_prod start_dev start_prod stop logs_dev logs_prod clean re
