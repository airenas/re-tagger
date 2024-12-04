####################################################################################
service=intelektikalt/ml-tagger-data
version?=dev
########### DOCKER ##################################################################
tag=$(service):$(version)

dbuild:
	@echo "Building docker image $(tag)"
	@echo "Version: $(version)"
	@echo "Data pack: $(data_pack)"
	cd .. && docker buildx build -t $(tag) --build-arg BUILD_VERSION=$(version) --build-arg DATA_PACK_DIR=$(data_pack) -f data_docker/Dockerfile .

dpush: dbuild
	docker push $(tag)

dscan: dbuild
	docker scan --accept-license $(tag)	
#####################################################################################
.PHONY: dbuild dpush
