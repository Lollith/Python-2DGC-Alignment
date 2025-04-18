DEV_COMPOSE=docker-compose.dev.yml
PROD_COMPOSE=docker-compose.prod.yml


all:  start_prod 

all_dev: build_dev start_dev 

pull_nist:
	@if [ -z "$$(docker images -q nist)" ]; then \
		echo "Image Docker NIST non trouvée. Construction en cours..."; \
		docker compose -f $(DEV_COMPOSE) build nist; \
	else \
		echo "Image Docker NIST déjà construite."; \
	fi


build_dev: 
	docker compose -f $(DEV_COMPOSE) build 2dgc_id


check_image:
	@if [ -z "$$(docker images -q 2dgc_id-app)" ]; then \
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

clean:
	sudo docker compose -f $(PROD_COMPOSE) down -v --rmi local
	sudo docker compose -f $(DEV_COMPOSE) down -v --remove-orphans --rmi local
	echo "Conteneurs, réseaux, images, volumes des environnements PROD et DEV supprimés."

clean_nist:
	sudo docker compose -f $(PROD_COMPOSE) down -v --rmi all nist
	sudo docker compose -f $(DEV_COMPOSE) down -v --remove-orphans --rmi all nist
	echo "Conteneurs, réseaux, images, volumes de l'environnement NIST supprimés."

re_dev: clean
	make all_dev

re: clean
	make all

.PHONY: all all_dev pull_nist check_image build_dev start_dev start_prod stop logs_dev logs_prod rebuild_prod clean clean_nist re re_dev
