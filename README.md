# Matrix Reduction Project in DPHPC Course
## Suggestion From TA
- Add a dace graph
- Comparison  
— Use absolute values & bandwidth (instead of runtime)  
— Compute standard deviation  
— Quantum chemistry: 1000 -> 10000  
— Upgrade cuda to 11.4 (check)  
— Uninstall and reinstall JAX  
— Run maybe 1000 times  
— Different sizes for 1D reduction  
— Mention competitor version
- Add pseudo-code for algorithms  
- P7, illustrate the algorithms
- P18, use diagram to show the scheduler logic
- Mention why atomic is faster than non-atomic for 2D & 3D
- Mention limitations on implementing high-dimension reduction  
## Link
google doc: https://docs.google.com/spreadsheets/d/1OkM_ZoOlsn_jmEJwMD0Ma14eUg3sMgGBJD9Qjom3mT8/edit#gid=859676680  

===================================================================================================
## PLAN
### 1.研究DaCe的使用
主要研究Dace的Mapping与生成代码的对应，如何生成我们想要的代码。
### 2.讨论二维算法
2.1 讨论不同输入大小应当使用哪种算法。（reasoning） 
这一部分思考完成之后也可以确定之后test所需要哪些input size。  
2.2 讨论[B,A]to[A]的情况。  
我们已经讨论过一种算法的基础模型：每个warp以cacheline为单位读数据，存到shared memory里面，再按列读取，求和。  
可以思考一下这个模型的细节，也可以思考一下，还有什么其它可能的算法或改进。  
这一部分内容需要记录下来，上传到GitHub，可以在meeting时讨论。  
2.3 在讨论一些implement细节的时候可以进行一些小型测试。  
### 3.库函数CODING
Prerequisite：1, 2.3, 2.2（partly） 
### 4.库函数测试
4.1 Test Script Coding，对不同input size测试不同的算法，找出每种size的最优算法。  
Prerequisite：2.1
4.2 Testing  
Prerequisite：2, 3, 4.1
### 5.Planner
5.1 Planner  
Prerequisite：4.2
5.2 Input Transfer  
Prerequisite：no  
将input转化为类似"ABA...BA","BAB...BA","ABA...AB","BAB...AB"的形式  
### 6.不同方法的测试比较
6.1 Study with potential competetion and write script  
Prerequisite：no  
6.2 Collect Real Data  
Prerequisite：no  
6.3 Test  
Prerequisite: all above  
### 7.High-dimension(not mandatory)
## Resources
DaCe Documentation: https://spcldace.readthedocs.io/en/latest/index.html  
DaCe GitHub: https://github.com/spcl/dace
