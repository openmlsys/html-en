# Privacy Encryption Algorithms

During federated learning, user data is used only for training on local
devices and does not need to be uploaded to the central FL-Server,
thereby preventing direct leakage of users' personal data. However, in
conventional federated learning frameworks, model weights are uploaded
to the cloud in plaintext, meaning that there is a risk of indirectly
leaking user privacy. Such leakage may occur because the plaintext
weights uploaded by users can be obtained by adversaries, who then
restore users' personal training data through reconstruction attacks or
model inversion attacks.

The MindSpore Federated framework provides the following algorithms to
add noise or perform scrambling before the weights of local models are
uploaded to the cloud: LDP- and MPC-based secure aggregation algorithms.
This helps prevent privacy leakage subject to the model availability
being guaranteed.

## Secure Aggregation Based on the LDP Algorithm

Differential privacy is a mechanism for protecting user data privacy.
Formula :eqref:`dp`
shows its definition.

$$Pr[\mathcal{K}(D)\in S] \le e^{\epsilon} Pr[\mathcal{K}(D') \in S]+\delta$$ 
:eqlabel:`eq:dp`

Given datasets $D$ and $D'$ that differ in one record only, the
probability of generating the $S$ subset by using the $\mathcal{K}$
random algorithm satisfies the preceding formula. $\epsilon$ indicates
the differential privacy budget, and $\delta$ indicates the
perturbation. Smaller values of $\epsilon$ and $\delta$ indicate that
the data distribution of $\mathcal{K}$ on $D$ is closer to that on $D'$.

Assuming that the model weight matrix obtained after local training on
the FL-Client is $W$, adversaries can utilize $W$ to restore users'
training dataset --- this is possible because the model memorizes the
characteristics of the training set during training.

MindSpore Federated provides LDP-based secure aggregation algorithms to
prevent private data leakage in the process of uploading the weights of
local models to the cloud.

The FL-Client generates a differential noise matrix $G$ that is of the
same dimension as the local model weight matrix $W$, and then adds the
two matrices together in order to obtain the $W_p$ weight matrix that
meets the definition of differential privacy, as shown in Formula
:eqref:`quanzhong`.

$$W_p=W+G$$ 
:eqlabel:`eq:quanzhong`

The FL-Client uploads the model weight matrix $W_p$ to the FL-Server on
the cloud for federated aggregation. Equivalent to the original model
with a mask layer, the noise matrix $G$ lowers the risk of leaking
sensitive data from the model but affects the convergence of model
training. Further studies are needed to determine how we can strike a
balance between model privacy and availability. In cases where there are
more than 1000 participants ($n$), experiments show that most of the
noise additions cancel each other out, and that the LDP mechanism has no
significant impact on the accuracy or convergence of aggregation models.

## Secure Aggregation Based on the MPC Algorithm

Although the differential privacy technology can adequately protect user
data privacy, the accuracy of the model is significantly compromised if
the number of participating FL-Clients is small or the amplitude of
Gaussian noise is large. To meet the requirements for both model
protection and model convergence, MindSpore Federated provides the
MPC-based secure aggregation solution.

Assuming that the set of participating FL-Clients is $U$ in such a
solution, every two FL-Clients ($u$ and $v$) can negotiate a pair of
random disturbance terms $p_{uv}$ and $p_{vu}$, which satisfy Formula
:eqref:`xieshang`.

$$    p_{uv}=
    \begin{cases}
    -p_{vu}, &u{\neq}v\\
    0, &u=v
    \end{cases}$$ 
:eqlabel:`eq:xieshang`

Each FL-Client ($u$) adds the disturbance obtained through negotiation
with other FL-Clients to the original model weight $x_u$ before
uploading the model weight to the FL-Server, as shown in Formula
:eqref:`qita`.

$$x_{encrypt}=x_u+\sum\limits_{v{\in}U}p_{uv}$$ 
:eqlabel:`eq:qita`

Formula :eqref:`juhejieguo` shows the aggregation result $\overline{x}$
on the FL-Server.

$$\overline{x}=\sum\limits_{u{\in}U}(x_{u}+\sum\limits_{v{\in}U}p_{uv})=\sum\limits_{u{\in}U}x_{u}+\sum\limits_{u{\in}U}\sum\limits_{v{\in}U}p_{uv}=\sum\limits_{u{\in}U}x_{u}$$ 
:eqlabel:`eq:juhejieguo`

Only the overall concept of aggregation algorithms is described above.
Although the MPC-based aggregation solution involves no loss of
accuracy, it increases the number of communication rounds.
