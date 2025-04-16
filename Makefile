DEV_COMPOSE=docker-compose.dev.yml
PROD_COMPOSE=docker-compose.prod.yml

all_dev: build_dev start_dev

all: start_prod

build_dev: 
	docker compose -f $(DEV_COMPOSE) build

check_image:
	@if [ -z "$$(docker images -q python-2dgc-alignment_prod)" ]; then \
		echo "Image Docker non trouvée. Construction en cours..."; \
		docker compose -f $(PROD_COMPOSE) build; \
	else \
		echo "Image Docker déjà construite."; \
	fi

start_dev :
	docker compose -f $(DEV_COMPOSE) up

start_prod: check_image
	docker compose -f $(PROD_COMPOSE) up

stop:
	- docker compose -f $(DEV_COMPOSE) down -v
	- docker compose -f $(PROD_COMPOSE) down -v

logs_dev:
	docker compose -f $(DEV_COMPOSE) logs -f

logs_prod:
	docker compose -f $(PROD_COMPOSE) logs -f

rebuild_prod:
	docker compose -f $(PROD_COMPOSE) build --no-cache
	echo "Image Docker reconstruite avec succès."

clean: stop
		sudo docker system prune -af --volumes
		echo "Nettoyage terminé."

re_dev: clean
	make all_dev

re: clean
	make all

.PHONY: all all_dev all_prod build_dev build_prod start_dev start_prod stop logs_dev logs_prod clean re
