# 十、并查集

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----


@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 并查集


> 并查集是一种**树型的数据结构**，用于处理一些**不相交集合的合并**及查询问题。
并查集的思想是用一个数组表示了整片森林（parent），树的**根节点唯一标识了一个集合**，我们只要找到了某个元素的的树根，就能确定它在哪个集合里。


* ==用途==: 处理不相交集合(disjointsets)的合并和查询问题 >> 处理分组问题 >> 维护无序二元关系


* ==实现==: 最简单的实现是只用一个int数组fa, fa[x]表示编号为x的结点的父结点 根结点的fa等于它自己


* 初始化
![在这里插入图片描述](https://img-blog.csdnimg.cn/9b59920b0b074b1bb89e451dfbbd9f1c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
* 合并
![在这里插入图片描述](https://img-blog.csdnimg.cn/3f7b983cbfc64a4d91937dc0067d1b77.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
* 查询
![在这里插入图片描述](https://img-blog.csdnimg.cn/f2eff23bcb5e43dfbf1fc49936725e88.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
* 并查集还有一个优化叫做**按秩合并**(合并时把**较深的树合并到较浅的上面**)或者启发式合并(合并 时把**较大的树合并到较小的树**上面) 
* 同时采用**路径压缩**+**按秩合并优化**的并查集，单次操作的均摊复杂度为`O(α(n))`
* 只采用其中一种，O(log(n)) 
* α(n)是反阿克曼函数，是一个比log(n)増长还要缓慢许多的函数，一般a(n) < 5


### 1.1相关题目
####  1.1.1 [547 . 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

* ==思路==:  对于相邻城市之间的边做合并 >>> 有几个根就有几个省

```python
class Union():
    def __init__(self,n):
        self.fa = [i for i in range(n)]
    def find(self, x):
        if self.fa[x] == x:return x
        self.fa[x] = self.find(self.fa[x])
        return self.fa[x]
    def unionSet(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            self.fa[x] = y
            
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n, ans = len(isConnected), 0
        un = Union(n)

        # 合并有边的两个集合
        for i in range(n):
            for j in range(n):
                if isConnected[i][j]:
                    un.unionSet(i, j)

        # 有几个根就有几个省
        for i in range(n):
            if un.find(i) == i:
                ans += 1
        return ans

```


####  1.1.2 [130 . 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)

* ==思路==:  在区域外部新建某个无限大区域outside  >>>  使得outside与四周的 'O' 相连  >>> 根为outside保留 


```python
class Union():
    def __init__(self,n):
        self.fa = [i for i in range(n)]
        self.outside = self.fa[-1]
    def num(i, j):
        return i
    def find(self, x):
        if self.fa[x] == x:return x
        self.fa[x] = self.find(self.fa[x])
        return self.fa[x]
    def unionSet(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            self.fa[x] = y

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        m, n = len(board), len(board[0])
        un = Union(n * m + 1)
        ouside = un.fa[-1]

        def num(i, j):
            return i * n + j

        dr = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'X':continue
                for k in dr:
                    ni = i + k[0]
                    nj = j + k[1]
                    if ni < 0 or nj < 0 or ni >= m or nj >= n:   # 若下一方向节点出界且为O 与outside相连
                        un.unionSet(num(i, j), un.outside)
                    else:
                        if board[ni][nj] == 'O':
                            un.unionSet(num(i,j), num(ni,nj))
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O' and un.find(num(i, j)) != un.find(un.outside): #与外部不相连且为O
                    board[i][j] = 'X'
        

        




```


## 2 图及相关算法



 链表、树、图的关系: **链表是特殊化的树  , 树是特殊化的图**

* 图的存储与添加元素:
  * 邻接矩阵  : 矩阵  O(n^2)
  * 出边数组  : 数组 + 数组 O(m + n)
  * 邻接表 : 数组 + 链表 O(m + n)

![在这里插入图片描述](https://img-blog.csdnimg.cn/8d5e7195b97340d7af70aac91290499f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.1  Bellman-Ford 算法


* 单源最短路径问题(Single Source Shortest Path, SSSP问题)是说：
  *  给定一张有向图`G = (V, E)`, V是点集，E是边集,`|V|=n, |E| =m` 
  * 节点以`［1，n］`之间的连续整数编号 
  * `(x,y, z)`描述一条从`x`出发，到达`y`,长度为`z`的有向边 
  * 设 `1 `号点为起点 
求长度为n的数组dist,其中`dist［i］`表示从起点1到节点i的最短路径的长度


* **Bellman-Ford算法**是基于**动态规划**和**迭代思想**的 
1 •扫描所有边`(x,y,z)`,若 `dist[y] > dist[x] + z`,则用 `dist[x] + z` 更新 `dist[y]`
2.重复上述步骤，直到没有更新操作发生
  *  **时间复杂度O(nm)** 
可以把每一轮看作DP的一个阶段 
第`i`轮至少已经求出了包含边数不超过`i`的最短路


---

* [743 . 网络延迟时间](https://leetcode-cn.com/problems/network-delay-time/)

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        dist = [1e9] * (n + 1)
        dist[k], ans = 0, 0
        for i in range(1, n):  #至多循环n-1轮
            updated = False
            for edge in times:
                x, y, z = edge
                if dist[y] > dist[x] + z: #比原先距离小则更新
                    dist[y] = dist[x] + z
                    updated = True
            if not updated:break

        for i in range(1, n + 1):
            ans = max(ans, dist[i])
        return -1 if ans == 1e9 else ans
```


### 2.2 Dijkstra 算法
* Dijkstra算法是基于**贪心思想**的，只适用于**所有边的长度都是非负数**的图 
 1. 初始化`dist[l] = 0`,其余节点的dist值为正无穷大。 
2. 找出一个未被标记的、dist[x]最小的节点`x`,然后标记节点`x`。 
3. 扫描节点 `x`的**所有出边(x, y, z)**,若 `dist[y] > dist[x] + z`,则使用 `dist[x] + z` 更新 `dist[y]`
4. 重复上述2〜3两个步骤，直到所有节点都被标记。


* 贪心思路：在非负权图上，**全局最小的dist值不可能再被其他节点更新**, 因此可以不断取dist最小的点(每个点只被取一次),更新其他点 
用二叉堆维护最小dist值可以做到O(m*log(n))的时间复杂度



1.  Dijkstra -- 懒惰删除   >>>>  O(n^2) + m

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        dist = [1e9] * (n + 1)
        dist[k], ans = 0, 0
        ver = [[] for _ in range(n + 1)]     #存点存边
        edge = [[] for _ in range(n + 1)]
        expand = [False] * (n + 1)
        for t in times:
            x, y, z = t
            ver[x].append(y)    #端点
            edge[x].append(z)   #边权


        for r in range(1 , n + 1):  #n轮
            temp = 1e9
            for i in range(1, n + 1):
                if not expand[i] and dist[i] < temp:  #没有拓展过  并且  目前最小dist[i]
                    temp = dist[i]
                    min_x = i            #从min_x出发考虑出边
            expand[min_x] = True

            for i in range(len(ver[min_x])):
                y = ver[min_x][i]
                z = edge[min_x][i]
                if dist[y] > dist[min_x] + z: 
                    dist[y] = dist[min_x] + z

        for i in range(1, n + 1):
            ans = max(ans, dist[i])
        return -1 if ans == 1e9 else ans
```



2. Dijkstra -- 堆 >> O(m*log(n))

```python
from heapq import *
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        dist = [1e9] * (n + 1)
        dist[k], ans = 0, 0
        ver = [[] for _ in range(n + 1)]     #存点存边
        edge = [[] for _ in range(n + 1)]
        expand = [False] * (n + 1)
        for t in times:
            x, y, z = t
            ver[x].append(y)    #端点
            edge[x].append(z)   #边权

        q = []
        heappush(q, (0, k))
        while q:
            distance, min_x = heappop(q)
            if expand[min_x]:continue  #最小已经拓展过
            expand[min_x] = True

            for i in range(len(ver[min_x])):
                y = ver[min_x][i]
                z = edge[min_x][i]
                if dist[y] > dist[min_x] + z: 
                    dist[y] = dist[min_x] + z
                    heappush(q, (dist[y], y))

        for i in range(1, n + 1):
            ans = max(ans, dist[i])
        return -1 if ans == 1e9 else ans
        
```


### 2.3 Floyd算法

* Floyd算法可以在`O(n^3)`时间内求出图中每一对点之间的最短路径 
本质上是动态规划算法 
* `dp[k, i,j]`表示经过**编号不超过k的点为中继**，从`i`到`j`的最短路 
决策：是否使用这个中继点 
`dp[k, i,j] = min(dp[k-1, i,j], dp[k - 1, i, k] + dp[k - 1, k, j]) `
可以省掉第一维，变成 
`d[i,j] = min(d[i,j], d[i, k] + d[k,j]) `
* 初态：d为邻接矩阵(原始图中的边) 
* 与 Bellman-Ford, Dijkstra 的比较：`O(n^3) vs O(n^2*m) vs O(nmlogn)`


[1334 . 阈值距离内邻居最少的城市](https://leetcode-cn.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)


```python
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        # 存邻接矩阵d
        d = [[1e9] * n for _ in range(n)]
        for i in range(n):
            d[i][i] = 0
        for edge in edges:
            x, y, z = edge
            d[x][y] = d[y][x] = z

        # dp  --  Floyd算法
        for k in range(n):#中继点  必须先阶段
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])  #用中继或不用

        # 统计neighbour
        Minneighbour, ans = 1e9, 0
        for i in range(n):
            neighbour = 0
            for j in range(n):
                if i != j and d[i][j] <= distanceThreshold:
                    neighbour += 1
            if Minneighbour > neighbour or (Minneighbour == neighbour and i > ans):
                Minneighbour = neighbour
                ans = i
        return ans
```




### 2.4 Kruskal算法

* Kruskal算法总是使用**并查集维护无向图的最小生成森林** 
1.建立并查集，每个点各自构成一个集合。 
2.把所有边按照权值从小到大排序，依次扫描每条边` (x, y, z)`
3.若`x,y`属于同一集合(连通)，则忽略这条边，继续扫描下一条。 
4. 否则，合并`x,y`所在的集合，并把`z`累加到答案中。 
5. 所有边扫描完成后，第4步中处理过的边就构成最小生成树。 时间复杂度为`O(mlogm)`



[1584 . 连接所有点的最小费用](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/)

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # 构造出边
        edges, n, ans = [], len(points), 0
        for i in range(n):
            for j in range(i + 1, n): #i到j 与 j 到 i相同
                edges.append([i, j, abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])])
        # 边权排序
        edges.sort(key = lambda x: x[2])
        # Kruskal算法
        self.fa = []
        for i in range(n):
            self.fa.append(i)

        for edge in edges:
            x, y, z = self.find(edge[0]), self.find(edge[1]), edge[2]
            if x != y:
                self.fa[x] = y
                ans += z
        return ans
        
    def find(self, x):
        if x == self.fa[x]:
            return x
        self.fa[x] = self.find(self.fa[x])
        return self.fa[x]
```


## 作业
### 1 [684 . 冗余连接](https://leetcode-cn.com/problems/redundant-connection/)
* 并查集解法

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        fa = [i for i in range(n + 1)]


        def find(x):
            if x == fa[x]:
                return x
            fa[x] = find(fa[x])
            return fa[x]

        def unionSet(x, y):
            x, y = find(x), find(y)
            if x == y:
                return True    #有环
            else:
                fa[y] = x
                return False   #目前没环

        for s, t in edges:
            if unionSet(s, t):
                return [s, t]
```


### 2 [200 . 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)
* 同样并查集

```python
class Union():
    def __init__(self,n):
        self.fa = [i for i in range(n)]
        self.outside = self.fa[-1]
    def find(self, x):
        if self.fa[x] == x:return x
        self.fa[x] = self.find(self.fa[x])
        return self.fa[x]
    def unionSet(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            self.fa[x] = y

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n, ans = len(grid), len(grid[0]), 0
        un = Union(m * n)

        def num(i, j):
            return i * n + j

        dr = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '0':continue
                for k in dr:
                    ni = i + k[0]
                    nj = j + k[1]
                    if ni < 0 or nj < 0 or ni >= m or nj >= n or grid[ni][nj] == '0':
                        continue
                    un.unionSet(num(i,j), num(ni,nj))
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and un.find(num(i, j)) == num(i, j):
                    ans += 1
        return ans
```

