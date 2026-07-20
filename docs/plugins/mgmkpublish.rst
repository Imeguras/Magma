mgmkpublish — Kafka Sink
=========================

Publishes ``MagmaInferenceMeta`` results to a Kafka topic.  Requires
``librdkafka`` at build time.

Pipeline position: ``… → mgminfer → mgmkpublish → …``

Properties
----------

.. doxygenfile:: mgmkpublish.hpp
   :project: Magma
