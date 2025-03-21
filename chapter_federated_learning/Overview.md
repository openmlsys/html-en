# Overview

With the rapid development of artificial intelligence (AI), large-scale
and high-quality data is playing an increasingly important role in
achieving optimum model effect and user experience. However, further
development of AI is restricted by a data utilization bottleneck,
whereby data cannot be shared among devices due to issues regarding
privacy, supervision, and engineering, resulting in data silos. To
resolve this data silo problem, the concept of federated learning was
proposed back in 2016. It aims to effectively utilize multi-party data
for machine learning modeling while also meeting the requirements of
user privacy protection, data security, and government regulations.

## Definition

Centralizing data from multiple parties means that user privacy
protection cannot be guaranteed --- such an approach would also fail to
comply with relevant laws and regulations. The core idea behind
federated learning is that models move whereas data stays put. It
enables models to move among data parties so that data can be used for
modeling without being transferred out of devices. In federated
learning, data of all parties is retained locally, and machine learning
models are established by exchanging encrypted parameters or other
information (on central servers).

## Application Scenarios

Federated learning can be classified into three categories based on
whether samples and features overlap: horizontal federated learning
(with different samples and overlapping features), vertical federated
learning (with different features and overlapping samples), and
federated transfer learning (without overlapping samples or features).

**Horizontal federated learning** applies to scenarios where different
individual participants have identical features. For example, in an
advertisement recommendation scenario, algorithm developers use data of
a specific feature (e.g., number of clicks, time on page/site, or
frequency of use) relating to different mobile phone users in order to
establish a model. Because such feature data cannot be transferred out
of devices, horizontal federated learning is used to establish models by
combining the feature data of multiple users.

**Vertical federated learning** applies to scenarios with many
overlapping samples but few overlapping features. Take two institutions
as an example: one is an insurance company and the other is a hospital.
The user groups of the two institutions are likely to include many local
residents, meaning that the two institutions may have a large
intersection of users. The insurance company holds data on users'
income, expense statements, and credit ratings, whereas the hospital
holds data on users' health and medical purchase records, resulting in a
small intersection of user features. Vertical federated learning
enhances model capabilities by aggregating different features in an
encrypted state.

**Federated transfer learning** aims to find the similarities between
the source and target fields. Take another two institutions as an
example: one is a bank in country A and the other is an e-commerce
company in country B. The user groups of the two institutions have a
small intersection due to geographical restrictions. In addition,
because the two institutions are dissimilar, only a small part of their
data features overlap. In this case, federated transfer learning is one
of the only ways to implement federated learning effectively and improve
the model effect because it can overcome the fact that there is limited
single-side data and few labeled samples.

## Deployment Scenarios

The architecture of federated learning is similar to that of a parameter
server (i.e., distributed learning in data centers). In both
architectures, centralized servers and distributed clients (i.e.,
multiple clients communicate with one server, and there is no
communication between clients) are used to build a machine learning
model. Based on the scenario in which it is deployed, federated learning
can be classified into cross-silo federated learning and cross-device
federated learning. Generally, users of cross-silo federated learning
are enterprises and institutions, whereas cross-device federated
learning is oriented to portable electronic devices (PEDs), mobile
devices, and the like. Table
[\[ch10-federated-learning-different-connection\]](#ch10-federated-learning-different-connection){reference-type="ref"
reference="ch10-federated-learning-different-connection"} describes the
differences and relationships among distributed learning in data
centers, cross-silo federated learning, and cross-device federated
learning.

[]{#ch10-federated-learning-different-connection
label="ch10-federated-learning-different-connection"}

## Common Frameworks

As users and developers continue to place higher demands on federated
learning technologies, more and more federated learning tools and
frameworks are emerging. The following lists some of the mainstream
federated learning frameworks:

1.  **TensorFlow Federated (TFF):** an open-source federated learning
    framework developed by Google to promote open research and
    experimentation in federated learning. It is used to implement
    machine learning and other types of computing on decentralized data.
    In this framework, a shared global model is trained among many
    participating customers who save their training data locally. For
    example, federated learning has been successfully used to train
    prediction models for mobile keyboards without uploading sensitive
    typed data to the server.

2.  **PaddleFL:** an open-source federated learning framework proposed
    by Baidu based on PaddlePaddle. It enables researchers to easily
    replicate and compare different federated learning algorithms and
    allows developers to readily deploy PaddleFL-based federated
    learning systems in large-scale distributed clusters. This framework
    provides multiple federated learning strategies (e.g., horizontal
    federated learning and vertical federated learning) and
    corresponding applications in fields such as computer vision,
    natural language processing, and recommendation algorithms. It also
    supports the application of traditional machine learning training
    strategies, for example, applying transfer learning to multitask
    learning and federated learning environments. PaddleFL can be easily
    deployed based on full-stack open-source software and leveraging
    PaddlePaddle's large-scale distributed training capability and
    Kubernetes' capability of elastically scheduling training tasks.

3.  **Federated AI Technology Enabler (FATE):** the world's first
    industrial-grade open-source framework proposed by WeBank for
    federated learning. It enables enterprises and institutions to
    collaborate on data while also ensuring data security and preventing
    data privacy leakage. By using secure multi-party computation (MPC)
    and homomorphic encryption technologies to build low-level secure
    computation protocols, FATE supports secure computation of different
    types of machine learning, including logistic regression, tree-based
    algorithms, deep learning, and transfer learning. This framework was
    opened to the public for the first time in February 2019 along with
    the launch of the FATE community, whose members include major cloud
    computing and financial service enterprises in China.

4.  **FedML:** an open-source research and baseline library proposed by
    the University of Southern California (USC) for federated learning.
    It facilitates the development of new federated learning algorithms
    and fair performance comparison. FedML supports three computing
    paradigms (i.e., distributed training, training on mobile devices,
    and independent simulation) for users to conduct experiments in
    different system environments. It also implements and promotes
    diversified algorithm research through flexible and general-purpose
    API design and reference baselines. To enable fair comparison of
    federated learning algorithms, FedML provides comprehensive
    benchmark datasets, including non-independent and identically
    distributed (non-iid) datasets.

5.  **PySyft:** a Python library released by University College London
    (UCL), DeepMind, and OpenMined for deep learning of security and
    privacy. It involves federated learning, and differential privacy
    (An encryption method: The differential privacy method is used to
    ensure that the impact of a single record on the data set is always
    lower than a certain threshold when the information is output, so
    that the third party cannot judge the change or deletion of a single
    record according to the change of the output. This method is
    considered as the highest security level in the current
    perturbation-based privacy protection method), and multi-party
    learning. PySyft uses differential privacy and encrypted computation
    (MPC and homomorphic encryption) to decouple private data from model
    training.

6.  **Fedlearner:** a vertical federated learning framework proposed by
    ByteDance for joint modeling based on data distributed among
    institutions. It comes with peripheral infrastructure for cluster
    management, job management, job monitoring, and network proxy.
    Fedlearner uses the cloud-native deployment solution and stores data
    in Hadoop Distributed File System (HDFS), and manages and starts
    tasks through Kubernetes. The two parties involved in a Fedlearner
    training task need to start the task simultaneously by using
    Kubernetes. All training tasks are managed by the master node in a
    unified manner, and the communication is implemented through Worker.

7.  **OpenFL:** a Python framework proposed by Intel for federated
    learning. OpenFL is designed to be a flexible, scalable, and
    easy-to-learn tool for data scientists.

8.  **Flower:** an open-source federated learning system released by the
    University of Cambridge for performing optimization in application
    scenarios where federated learning algorithms are deployed on
    large-scale heterogeneous devices.

9.  **MindSpore Federated:** an open-source federated learning framework
    proposed by Huawei. It supports the commercial deployment of tens of
    millions of stateless devices, and enables all-scenario intelligent
    applications when user data is stored locally. MindSpore Federated
    focuses on horizontal federated learning involving a large number of
    participants, enabling them to jointly build AI models without
    sharing local data. It mainly addresses the difficulties of
    deploying federated learning in industrial scenarios, including
    difficulties in privacy security, large-scale federated aggregation,
    semi-supervised federated learning, communication compression, and
    cross-platform deployment.
