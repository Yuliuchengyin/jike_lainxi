# 四、树图, 搜索

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----


@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 树与图
### 1.1 二叉树遍历

>* 二叉树的**遍历方式**主要有：**先序遍历、中序遍历、后序遍历、层次遍历**。
>>  * 先序、中序、后序其实指的是**父节点被访问的次序**。若在遍历过程中，**父节点先于**它的子节点被访问，就是先序遍历；**父节点被访问**的次序位于左右孩子节点之间，就是中序遍历；访问完左右孩子节点之后**再访问父节点**，就是后序遍历。不论是先序遍历、中序遍历还是后序遍历，左右孩子节点的相对访问次序是不变的，总是**先访问左孩子节点，再访问右孩子节点**。
  >>* 层次遍历，就是按照从**上到下、从左向右**的顺序访问二叉树的每个节点。



* python 中定义二叉树

```python
class TreeNode: 
	def init(selfval): 
		self.val = val 
		self.leftself.right = None, None
```

* **前序Pre-order:根-左子树-右子树:**
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/afc8ba31b885491bac2d6e532a274e8e.png)
 
* 动图演示

![请添加图片描述](https://img-blog.csdnimg.cn/4311db7fea5244f8889794c158731b9c.gif)


---


* **中序In-order:左子树-根-右子树**
![在这里插入图片描述](https://img-blog.csdnimg.cn/b6c1e7bef6804887b66e81dda1654b23.png)

* 动图:

![请添加图片描述](https://img-blog.csdnimg.cn/c20fe0d6e7be4f3cbf308f8b10004fec.gif)


---
* **后序Post-order:左子树-右子树-根**
![在这里插入图片描述](https://img-blog.csdnimg.cn/f0d044aad7294189b9ee114e5ea07429.png)


* 动图:


![请添加图片描述](https://img-blog.csdnimg.cn/6ac690e9f16c4e3298dd69cdfdf62b0d.gif)

---

* **层次序**:
![在这里插入图片描述](https://img-blog.csdnimg.cn/16ea494b20b34e27a8eef1d0c3c0336e.png)


![在这里插入图片描述](https://img-blog.csdnimg.cn/5b7eb2a3a70d417ea197b2510cc84c32.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



* 先序、中序、后序一般用==递归==来求 ; 树的先序遍历又称**树的深度优先遍历** 
* 层次序一般借助==队列==来求 ; 树的层序遍历又称树的**广度优先遍历**

### 1.2 相关例题

#### 1.2.1 [94 . 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)


* ==思路==:   深度 >>>  递归 >>> 中:  左递归   根加入    右递归

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        ans = []
        self.dfs(root, ans)
        return ans

    def dfs(self, root, ans):
        if not root: return 
        self.dfs(root.left, ans)
        ans.append(root.val)
        self.dfs(root.right, ans)
```




#### 1.2.2 [589 . N 叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)

* ==思路一== : 递归 >>>  根: 录入  子节点: 递归


```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        ans = []
        self.dfs(root, ans)
        return ans

    def dfs(self, root, ans):
        if  not root: return
        ans.append(root.val)
        for i in root.children:
            self.dfs(i, ans)
```




* ==思路二== : 迭代 >>>  **栈**模拟递归   >>>  子节点从右至左放入栈


```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        stack, ans = [], []
        if not root: return []
        stack.append(root)  #根节点入栈
        
        while stack:
            r = stack.pop()   #根节点出栈
            ans.append(r.val) 
            for i in range(len(r.children) - 1, -1, -1): #子节点倒序入栈
                stack.append(r.children[i])

        return ans
```



#### 1.2.3 [429 . N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)
* ==思路== : 层次  >>> 队列迭代 >>> 


```python
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        ans, que = [], []
        if not root: return ans
        que.append(root)         # 加入根节点[1]

        while que:
            depth = []
            for _ in range(len(que)):    #遍历depth所在层次
                tmp = que.pop(0)         # 按顺序输出子节点
                depth.append(tmp.val)
                que.extend(tmp.children)    # 加入子节点
            ans.append(depth)

        return ans
```





#### 1.2.4 [105 . 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)


* ==思路一==: 递归构建左 右  根>>>>  根为 `root = preorder[ 0 ]`   在 `inorder`中`[left, root - 1]`为左子树中序; `[root + 1, right]`为右子树中序;  其中, `n = root - 1 - left + 1`为左子树元素个数根据个数求得`preorder [1: n]`为左子树前序   >>>  再对子树递归

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def build_new(l1, r1, l2, r2):
            if r1 < l1:return 
            root = TreeNode(preorder[l1])
            mid = l2 
            while inorder[mid] != root.val:
                mid += 1      
            root.left = build_new(l1 + 1, l1 + (mid - l2), l2, mid - 1)
            root.right = build_new(l1 + (mid - l2) + 1, r1, mid + 1, r2)
            return root

        m, n = len(preorder) - 1, len(inorder) - 1
        return build_new(0, m, 0, n)
```




* ==思路二== >>> 同样递归, 但python中可用**索引寻找值**以及**通过切片完成区间选取**  >>> 更为简洁





```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder: return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])  # 用preorder[0]去中序数组中查找对应的元素
        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])
        return root
```

#### 1.2.5 [297 . 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)
* 思路:  
1. 先想到  :  (1)序列化:  求前中序以string存   (2)反序列化:  利用两个序列构造二叉树  >>>  节点值可能右重复 >>> 不可以像105题直接记下`mid` 即为`root`  
  2. 拓展先序 >>> 无子节点存null


```python
class Codec:

    def serialize(self, root):
        seq = []
        def dfs(seq, root):
            if not root: 
                seq.append("None")         # 空处加null
                return
            seq.append(str(root.val))      #求前序
            dfs(seq, root.left)  
            dfs(seq, root.right)
            return seq
        dfs(seq, root)
        return ",".join(seq)       

    def deserialize(self, data):
        seq = data.split(",")
        def restore():
            temp = seq.pop(0)
            if temp == "None":
                return None
            root = TreeNode(int(temp))
            root.left = restore()
            root.right = restore()
            return root
        return restore()
```





### 1.3 图的遍历



* 链表、树、图的关系: **链表是特殊化的树  , 树是特殊化的图**

* 图的存储与添加元素:
  * 邻接矩阵  : 矩阵  O(n^2)
  * 出边数组  : 数组 + 数组 O(m + n)
  * 邻接表 : 数组 + 链表 O(m + n)

![在这里插入图片描述](https://img-blog.csdnimg.cn/8d5e7195b97340d7af70aac91290499f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



* 图的遍历

* 深度优先遍历
   （1 从图中的某个初始点 v 出发，首先访问初始点 v.
（2 选择一个与顶点 v 相邻且没被访问过的顶点 ver ，以 ver 为初始顶点，再从它出发进行深度优先遍历。
（3 当路径上被遍历完，就访问上一个顶点的第 二个相邻顶点。
（4 直到所有与初始顶点 v 联通的顶点都被访问。


![在这里插入图片描述](https://img-blog.csdnimg.cn/e31c3f6853e4449aa460f03882285abc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


---

* 广度优先遍历
（1）从图中的某个初始点 v0 出发，首先访问初始点 v0。
（2）接着访问该顶点的所有未访问过的邻接点 v01 v02 v03 ……v0n。
（3）然后再选择 v01 v02 v03 ……v0n，访问它们的未被访问的邻接点，v010 v011 v012……v01n。
（4）直到所有与初始顶点 v 联通的顶点都被访问。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b1a28f3af39946f08d66842f66456ca8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)




---


#### 1.4 相关题目
#### 1.4.1 [684 . 冗余连接](https://leetcode-cn.com/problems/redundant-connection/)

* 思路:  寻找基环树的环在哪? >>> 深度遍历  >>>> 

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n, x, y = 0, 0, 0
        for edge in edges:    # 计算出n
            x = edge[0]
            y = edge[1]
            n = max(n, max(x, y))
        visvited = [False for i in range(n + 1)]
        to = [[] for i in range(n + 1)]        # 出边数组存图

        def dfs(x,father):
            global hascycle
            visvited[x] = True
            for y in to[x]:
                if y == father: continue  # 双向边不为环
                if not visvited[y]: dfs(y, x)  # x 是y的父
                else:
                    hascycle = True          # 既不是双向边又遍历过 > 即是环


        for edge in edges:      # 找所有双向边储存
            x = edge[0]
            y = edge[1]
            to[x].append(y)
            to[y].append(x)
            hascycle = False
            visvited = [False for i in range(n + 1)]
            dfs(x, 0)
            if dfs(x, 0):
                return edge
```


#### 1.4.2 [207 . 课程表](https://leetcode-cn.com/problems/course-schedule/)

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        to = [[] for i in range(numCourses)]
        inDeg = [0 for i in range(numCourses)]
        for pre in prerequisites:
            ai = pre[0]
            bi = pre[1]
            to[bi].append(ai)
            inDeg[ai] += 1
        
        q = []
        # 拓扑排序第一步: 从0入点出发
        for i in range(numCourses):
            if inDeg[i] == 0:       #无先修课
                q.append(i)
        lesson = []

        while len(q):
            x = q[0]
            q.pop(0)
            lesson.append(x)   #修x考虑后继课
            # 第二步: 扩展第一个点, 周围入点减一
            for y in to[x]:
                inDeg[y] -= 1
                # 第三步 入度减少为0 ,入队
                if inDeg[y] == 0:
                    q.append(y)
        return len(lesson) == numCourses



```

## 2 习题



### 2.1 [210 . 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        to = [[] for i in range(numCourses)]
        inDeg = [0 for i in range(numCourses)]
        for pre in prerequisites:
            ai = pre[0]
            bi = pre[1]
            to[bi].append(ai)
            inDeg[ai] += 1
        q = []
        for i in range(numCourses):
            if inDeg[i] == 0:       #无先修课
                q.append(i)
        lesson = []

        while len(q):
            x = q[0]
            q.pop(0)
            lesson.append(x)   #修x考虑后继课
            for y in to[x]:
                inDeg[y] -= 1
                if inDeg[y] == 0:
                    q.append(y)

        if len(lesson) == numCourses:
            return lesson
        else:
            return []
```


### 2.2 [106 . 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)


```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:

        if not postorder: return None
        root = TreeNode(postorder[-1])
        mid = inorder.index(postorder[-1])  # 用postorder[-1]去中序数组中查找对应的元素
        root.left = self.buildTree(inorder[:mid], postorder[: mid])
        root.right = self.buildTree(inorder[mid + 1:], postorder[mid:-1])
        return root
```

## 参考资料
1. <https://zhuanlan.zhihu.com/p/56895993>  二叉树的遍历详解
2. <https://blog.csdn.net/weixin_45525272/article/details/105837185>二叉树三种遍历（动态图+代码深入理解）
