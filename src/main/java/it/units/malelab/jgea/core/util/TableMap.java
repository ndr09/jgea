package it.units.malelab.jgea.core.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * @author andrea on 2021/06/17 for jgea
 */
public class TableMap<T> implements Table<T> {

    private final int size;
    private final List<T> values;

    public TableMap(int size) {
        this.size = size;
        this.values = new ArrayList<>(size*size);
        for(int i=0;i<size*size;i++){
            values.add(null);
        }
    }

    @Override
    public List<T> row(int y) {
        int nColumns = nColumns();
        return values.subList(y * nColumns, (y + 1) * nColumns);
    }

    @Override
    public int nRows() {
        return size;
    }

    @Override
    public int nColumns() {
        return size;
    }

    @Override
    public void set(int x, int y, T t) {
        checkIndexes(x, y);
        values.set(index(x, y), t);
    }

    @Override
    public T get(int x, int y) {
        checkIndexes(x, y);
        return values.get(index(x, y));
    }

    @Override
    public void addColumn(String name, List<T> values) {

    }

    @Override
    public void addRow(List<T> values) {

    }

    @Override
    public void clear() {
        values.clear();
    }

    @Override
    public List<String> names() {
        return null;
    }

    private int index(int x, int y) {
        return y * nColumns() + x;
    }

    public List<T> values(){
        return values.stream().filter(Objects::nonNull).collect(Collectors.toList());
    }

}
