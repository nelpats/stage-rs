Collaborative filtering builds recommendations based on the knowledge of users' attitudes towards items, that is, it uses the "wisdom of the crowd" to recommend items. It operates under the assumption that if a user $A$ has the same opinion as a user $B$ on an issue, $A$ is more likely to have $B$'s opinion on a different issue than that of a random user. There are two main types of collaborative filtering:

- **User-based collaborative filtering:** This method finds users that are similar to the target user (based on similarity in their ratings) and recommends items those similar users have liked. It requires calculating the similarity between users, which can be computationally intensive for large datasets.
    
- **Item-based collaborative filtering:** Instead of measuring the similarity between users, this method calculates the similarity between items based on users' ratings of those items. It recommends items that are similar to items the user has already liked or interacted with.

This type of recommander system can use [Knowledge Graphs](Knowledge%20Graphs.md) To find similar items/users. Another way could be to use [[Embedded Systems]]